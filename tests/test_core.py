"""Тесты ядра на синтетической стенограмме — без аудио и без интернета."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from callqa import checks as ck, metrics as mt, report
from callqa.config import Config
from callqa.models import CLIENT, OPERATOR, Segment, Transcript, Word
from callqa.pipeline import analyze_file
from callqa.text import SpeechIndex, find_pattern


def seg(spk, start, text, wps=2.6):
    words_txt = text.split()
    step = 1.0 / wps
    words, t = [], start
    for w in words_txt:
        words.append(Word(w, round(t, 2), round(t + step, 2)))
        t += step
    return Segment(spk, start, round(t, 2), text, words)


def demo_transcript() -> Transcript:
    """Разговор, в котором намеренно есть и хорошее, и плохое."""
    segs = [
        seg(OPERATOR, 0.0, "Здравствуйте меня зовут Игорь компания Светотехника чем могу помочь"),
        seg(CLIENT, 7.0, "Добрый день я хотел бы купить настольную лампу для работы"),
        seg(OPERATOR, 12.5, "Подскажите пожалуйста для каких целей вам нужна лампа и какой бюджет"),
        seg(CLIENT, 18.0, "Ну я работаю за компьютером вечерами хотелось бы что-то не очень дорогое"),
        # перебивание: оператор вклинивается за 1.5 с до конца реплики клиента
        seg(OPERATOR, 22.0, "Понял вас у нас есть отличная модель она стоит 4500 рублей"),
        seg(CLIENT, 28.5, "А что там по гарантии"),
        seg(OPERATOR, 31.0,
            "Стандартная гарантия год но я рекомендую оформить расширенную гарантию на два года"),
        seg(CLIENT, 38.0, "Дороговато выходит я подумаю"),
        seg(OPERATOR, 41.5,
            "Понимаю ваши сомнения зато вы будете спокойны два года и ремонт будет бесплатным"),
        seg(CLIENT, 48.5, "Хорошо а доставка есть"),
        seg(OPERATOR, 51.0, "Конечно доставка курьером завтра до двери оформляем заказ"),
        seg(CLIENT, 56.5, "Давайте оформляйте"),
        seg(OPERATOR, 59.0, "Отлично записываю вас спасибо за обращение хорошего дня до свидания"),
    ]
    return Transcript(segments=segs, duration=65.0, channels=2, source="stereo")


class TestText(unittest.TestCase):
    def setUp(self):
        self.tr = demo_transcript()
        self.op = SpeechIndex(self.tr.by_speaker(OPERATOR))

    def test_exact(self):
        ev = find_pattern(self.op, "меня зовут")
        self.assertIsNotNone(ev)
        self.assertEqual(ev.how, "exact")
        self.assertLess(ev.start, 5)

    def test_lemma(self):
        # в тексте «расширенную гарантию», ищем начальную форму
        ev = find_pattern(self.op, "расширенная гарантия")
        self.assertIsNotNone(ev)
        self.assertIn(ev.how, ("exact", "lemma", "fuzzy"))

    def test_regex(self):
        ev = find_pattern(self.op, r"re:\b\d[\d\s]{2,}\s*(руб|рубл)")
        self.assertIsNotNone(ev)
        self.assertEqual(ev.how, "regex")

    def test_not_found(self):
        self.assertIsNone(find_pattern(self.op, "ипотечное кредитование под залог"))

    def test_speaker_isolation(self):
        cl = SpeechIndex(self.tr.by_speaker(CLIENT))
        self.assertIsNotNone(find_pattern(cl, "я подумаю"))
        self.assertIsNone(find_pattern(self.op, "я подумаю"))

    def test_time_window(self):
        early = self.op.slice_by_time(None, 10)
        self.assertIsNotNone(find_pattern(early, "здравствуйте"))
        late = self.op.slice_by_time(30, None)
        self.assertIsNone(find_pattern(late, "здравствуйте"))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.load()
        self.tr = demo_transcript()

    def test_interruption_detected(self):
        m = mt.compute(self.tr, self.cfg)
        self.assertGreaterEqual(m["operator_interruptions"], 1)
        ev = m["_interruption_events"][0]
        self.assertEqual(ev["by"], OPERATOR)

    def test_backchannel_not_counted(self):
        tr = Transcript(segments=[
            seg(OPERATOR, 0.0, "Значит по вашему заказу мы отправим его завтра утром курьером"),
            seg(CLIENT, 3.0, "угу"),
        ], duration=12.0)
        m = mt.compute(tr, self.cfg)
        self.assertEqual(m["client_interruptions"], 0)

    def test_talk_ratio(self):
        m = mt.compute(self.tr, self.cfg)
        self.assertTrue(0 < m["operator_talk_ratio"] < 1)
        self.assertAlmostEqual(
            m["operator_talk_ratio"] + m["client_talk_ratio"], 1.0, places=2)

    def test_stop_phrases(self):
        tr = Transcript(segments=[
            seg(OPERATOR, 0.0, "Я не знаю это не моя работа обращайтесь в другой отдел"),
            seg(CLIENT, 6.0, "Ну спасибо"),
        ], duration=12.0)
        m = mt.compute(tr, self.cfg)
        self.assertGreaterEqual(m["stop_phrase_hits"], 2)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.load()
        self.cfg.llm.enabled = False       # тесты не ходят в сеть
        self.res = analyze_file("demo_call.wav", self.cfg,
                                transcript_override=demo_transcript())

    def test_ok(self):
        self.assertTrue(self.res.ok, self.res.error)

    def test_positive_checks(self):
        by = {c.id: c for c in self.res.checks}
        for cid in ("greeting", "self_intro", "needs_questions", "price_named",
                    "next_step", "farewell"):
            self.assertTrue(by[cid].passed, f"{cid} должен быть найден")

    def test_negative_check(self):
        by = {c.id: c for c in self.res.checks}
        self.assertFalse(by["ask_client_name"].passed, "имя клиента не спрашивали")

    def test_invert_check(self):
        by = {c.id: c for c in self.res.checks}
        self.assertTrue(by["no_argue"].passed, "оператор не грубил")

    def test_products(self):
        offered = {p.name: p.offered for p in self.res.products}
        self.assertTrue(offered["Расширенная гарантия"])
        self.assertTrue(offered["Доставка"])
        self.assertFalse(offered["Страховка"])

    def test_default_checklist_needs_no_llm(self):
        """Штатный чек-лист должен полностью считаться без языковой модели."""
        na = [c.title for c in self.res.checks if c.passed is None]
        self.assertEqual(na, [], f"без модели остались непроверенными: {na}")

    def test_llm_checks_are_na_without_key(self):
        """Если пункт всё-таки типа llm, без ключа он «н/д», а не «было»."""
        data = Config.load().to_dict()
        data["checks"] = [{"id": "smart", "title": "Смысловой", "type": "llm",
                           "speaker": "operator", "weight": 5,
                           "prompt": "Оператор был вежлив?"}]
        cfg = Config.from_dict(data)
        cfg.llm.enabled = True
        cfg.llm.api_key = ""
        cfg.llm.api_key_env = "CALLQA_НЕТ_ТАКОЙ_ПЕРЕМЕННОЙ"
        cfg.llm.base_url = ""
        res = analyze_file("x.wav", cfg, transcript_override=demo_transcript())
        self.assertIsNone(res.checks[0].passed)
        self.assertEqual(res.checks[0].status_ru, "н/д")

    def test_score(self):
        self.assertGreater(self.res.score_pct, 0)
        self.assertLessEqual(self.res.score_pct, 100)
        self.assertTrue(self.res.grade)

    def test_evidence_has_timecodes(self):
        by = {c.id: c for c in self.res.checks}
        ev = by["greeting"].evidence[0]
        self.assertGreaterEqual(ev.start, 0)
        self.assertTrue(ev.quote)


def cfg_with_check(check: dict) -> Config:
    data = Config.load().to_dict()
    data["checks"] = [check]
    data["products"] = []
    cfg = Config.from_dict(data)
    cfg.llm.enabled = False
    # короткие тестовые диалоги не должны отсеиваться фильтрами
    cfg.filters.min_words = 1
    cfg.filters.min_duration_sec = 0
    return cfg


class TestSequenceCheck(unittest.TestCase):
    """«Клиент возразил — оператор отреагировал»: замена LLM-проверке."""

    BASE = {"id": "obj", "title": "Отработал возражение", "type": "sequence",
            "speaker": "operator", "weight": 8, "within_sec": 45,
            "if_no_trigger": "pass", "trigger_speaker": "client",
            "trigger_any_of": ["дорого", "я подумаю"],
            "any_of": ["понимаю ваши сомнения", "зато", "давайте посчитаем"]}

    def run_on(self, tr, **over):
        cfg = cfg_with_check({**self.BASE, **over})
        return analyze_file("x.wav", cfg, transcript_override=tr).checks[0]

    def test_objection_answered(self):
        r = self.run_on(demo_transcript())
        self.assertTrue(r.passed, r.detail)
        self.assertIn("0:38", r.detail)

    def test_objection_ignored(self):
        tr = Transcript(segments=[
            seg(CLIENT, 0.0, "Дороговато выходит я подумаю"),
            seg(OPERATOR, 5.0, "Ну как хотите всего доброго до свидания"),
        ], duration=20.0)
        r = self.run_on(tr)
        self.assertFalse(r.passed)
        self.assertIn("реакции", r.detail)

    def test_no_trigger_passes_by_default(self):
        tr = Transcript(segments=[
            seg(CLIENT, 0.0, "Отлично беру спасибо большое"),
            seg(OPERATOR, 4.0, "Оформляю заказ хорошего дня"),
        ], duration=15.0)
        r = self.run_on(tr)
        self.assertTrue(r.passed)
        self.assertIn("не применяется", r.detail)

    def test_no_trigger_can_be_na(self):
        tr = Transcript(segments=[seg(CLIENT, 0.0, "Отлично беру спасибо")],
                        duration=10.0)
        r = self.run_on(tr, if_no_trigger="na")
        self.assertIsNone(r.passed)

    def test_response_outside_window_fails(self):
        tr = Transcript(segments=[
            seg(CLIENT, 0.0, "Это дорого для меня"),
            seg(OPERATOR, 3.0, "Угу"),
            seg(OPERATOR, 200.0, "Зато вы будете спокойны два года"),
        ], duration=220.0)
        r = self.run_on(tr, within_sec=30)
        self.assertFalse(r.passed, "ответ через три минуты не должен засчитываться")
        self.assertIn("реакции", r.detail)

    def test_response_must_come_from_right_speaker(self):
        """Фраза-ответ в устах клиента не считается реакцией оператора."""
        tr = Transcript(segments=[
            seg(CLIENT, 0.0, "Это дорого для меня"),
            seg(CLIENT, 4.0, "Хотя зато качество наверное хорошее"),
        ], duration=20.0)
        r = self.run_on(tr)
        self.assertFalse(r.passed, "реплика клиента не должна засчитываться "
                                   "как реакция оператора")

    def test_trigger_word_forms(self):
        """Возражение «дороговато» не совпадает с «дорого» — нужен свой вариант."""
        tr = Transcript(segments=[
            seg(CLIENT, 0.0, "Дороговато выходит"),
            seg(OPERATOR, 4.0, "Зато вы будете спокойны"),
        ], duration=20.0)
        self.assertTrue(self.run_on(tr, if_no_trigger="fail").passed is False,
                        "«дорого» не должно ловить «дороговато»")
        r = self.run_on(tr, trigger_any_of=["дороговато"])
        self.assertTrue(r.passed, r.detail)

    def test_empty_config_is_na(self):
        r = self.run_on(demo_transcript(), trigger_any_of=[], any_of=[])
        self.assertIsNone(r.passed)


class TestMinHits(unittest.TestCase):
    """min_hits: разовая обмолвка не должна засчитываться за системную работу."""

    BASE = {"id": "pol", "title": "Вежливость", "type": "keyword",
            "speaker": "operator", "weight": 5,
            "any_of": ["пожалуйста", "спасибо"]}

    def run_on(self, text: str, min_hits: int):
        tr = Transcript(segments=[seg(OPERATOR, 0.0, text)], duration=30.0)
        cfg = cfg_with_check({**self.BASE, "min_hits": min_hits})
        return analyze_file("x.wav", cfg, transcript_override=tr).checks[0]

    def test_not_enough(self):
        r = self.run_on("Подскажите пожалуйста номер заказа", 3)
        self.assertFalse(r.passed)
        self.assertIn("1", r.detail)

    def test_enough(self):
        r = self.run_on("Пожалуйста подождите спасибо за ожидание "
                        "и спасибо что дождались пожалуйста", 3)
        self.assertTrue(r.passed, r.detail)

    def test_min_hits_one_is_default_behaviour(self):
        r = self.run_on("Подскажите пожалуйста номер", 1)
        self.assertTrue(r.passed)

    def test_counts_repeats_of_same_phrase(self):
        r = self.run_on("спасибо спасибо спасибо", 3)
        self.assertTrue(r.passed, r.detail)


class TestOccurrences(unittest.TestCase):
    def setUp(self):
        from callqa.text import SpeechIndex

        tr = Transcript(segments=[
            seg(OPERATOR, 0.0, "спасибо вам спасибо большое и ещё раз спасибо"),
        ], duration=20.0)
        self.idx = SpeechIndex(tr.segments)

    def test_finds_every_occurrence(self):
        from callqa.text import find_occurrences

        hits = find_occurrences(self.idx, "спасибо")
        self.assertEqual(len(hits), 3)

    def test_occurrences_are_ordered_and_distinct(self):
        from callqa.text import find_occurrences

        hits = find_occurrences(self.idx, "спасибо")
        starts = [h.start for h in hits]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(len(set(starts)), 3)

    def test_count_across_patterns(self):
        from callqa.text import count_occurrences

        n, hits = count_occurrences(self.idx, ["спасибо", "большое"])
        self.assertEqual(n, 4)
        self.assertEqual(len(hits), 4)


class TestLocalLLMConfig(unittest.TestCase):
    """Локальный сервер должен работать без платного ключа."""

    def _cfg(self, base_url: str) -> Config:
        cfg = Config.load()
        cfg.llm.provider = "openai"
        cfg.llm.base_url = base_url
        cfg.llm.api_key = ""
        cfg.llm.api_key_env = "CALLQA_НЕТ_ТАКОЙ_ПЕРЕМЕННОЙ"
        return cfg

    def test_local_url_detected(self):
        for url in ("http://localhost:11434/v1", "http://127.0.0.1:1234/v1"):
            self.assertTrue(self._cfg(url).llm.is_local(), url)

    def test_remote_url_not_local(self):
        self.assertFalse(self._cfg("https://api.openai.com/v1").llm.is_local())

    def test_local_needs_no_key(self):
        self.assertEqual(self._cfg("http://localhost:11434/v1").llm_api_key(), "local")

    def test_remote_still_needs_key(self):
        self.assertEqual(self._cfg("https://api.openai.com/v1").llm_api_key(), "")

    def test_compact_prompt_auto_for_local(self):
        self.assertTrue(self._cfg("http://localhost:11434/v1").llm.use_compact())
        self.assertFalse(self._cfg("https://api.openai.com/v1").llm.use_compact())

    def test_compact_can_be_forced(self):
        cfg = self._cfg("https://api.openai.com/v1")
        cfg.llm.compact = "true"
        self.assertTrue(cfg.llm.use_compact())

    def test_base_url_gets_v1_suffix(self):
        from callqa.llm import normalize_base_url

        self.assertEqual(normalize_base_url("http://localhost:11434"),
                         "http://localhost:11434/v1")
        self.assertEqual(normalize_base_url("http://localhost:11434/v1"),
                         "http://localhost:11434/v1")
        self.assertEqual(normalize_base_url("http://localhost:11434/v1/"),
                         "http://localhost:11434/v1")

    def test_compact_answer_is_understood(self):
        """Плоский ответ маленькой модели должен разворачиваться в общий формат."""
        from callqa.llm import _expand_compact

        out = _expand_compact({"answers": [
            {"id": "politeness", "passed": True, "quote": "спасибо"},
            {"id": "товар::Страховка", "passed": False, "quote": ""},
        ]})
        self.assertEqual(len(out["checks"]), 1)
        self.assertEqual(out["checks"][0]["id"], "politeness")
        self.assertEqual(out["products"][0]["name"], "Страховка")
        self.assertFalse(out["products"][0]["offered"])


class TestComputeType(unittest.TestCase):
    """На Pascal (GTX 10xx) float16 не поддерживается — нельзя его запрашивать."""

    def test_pascal_gets_int8_float32(self):
        from callqa import asr

        orig = asr.compute_capability
        try:
            asr.compute_capability = lambda: 6.1
            self.assertEqual(asr._pick_compute_type("auto", "cuda"), "int8_float32")
        finally:
            asr.compute_capability = orig

    def test_turing_and_newer_get_float16(self):
        from callqa import asr

        orig = asr.compute_capability
        try:
            for cc in (7.5, 8.6, 9.0):
                asr.compute_capability = lambda c=cc: c
                self.assertEqual(asr._pick_compute_type("auto", "cuda"), "float16")
        finally:
            asr.compute_capability = orig

    def test_unknown_gpu_gets_safe_default(self):
        from callqa import asr

        orig = asr.compute_capability
        try:
            asr.compute_capability = lambda: None
            self.assertEqual(asr._pick_compute_type("auto", "cuda"), "int8_float32")
        finally:
            asr.compute_capability = orig

    def test_explicit_setting_wins(self):
        from callqa import asr

        self.assertEqual(asr._pick_compute_type("float32", "cuda"), "float32")

    def test_cpu_uses_int8(self):
        from callqa import asr

        self.assertEqual(asr._pick_compute_type("auto", "cpu"), "int8")


class TestReport(unittest.TestCase):
    def setUp(self):
        self.cfg = Config.load()
        self.cfg.llm.enabled = False
        self.res = [analyze_file("demo_call.wav", self.cfg,
                                 transcript_override=demo_transcript())]

    def test_summary_shape(self):
        header, rows, statuses = report.summary_rows(self.res, self.cfg)
        self.assertEqual(len(header), len(rows[0]))
        self.assertEqual(len(header), len(statuses[0]))

    def test_summary_shows_phrase_and_timecode(self):
        """В ячейке должна быть найденная фраза и секунда, а не «было»."""
        header, rows, _st = report.summary_rows(self.res, self.cfg)
        cells = dict(zip(header, rows[0]))
        greeting = cells["Поздоровался"]
        self.assertIn("здравствуйте", greeting.lower())
        self.assertRegex(greeting, r"\d+:\d\d", "нет таймкода")
        self.assertNotIn("было", greeting)

    def test_regex_cell_shows_matched_text_not_pattern(self):
        """Для регулярки показываем совпавший текст, а не сам шаблон."""
        _h, rows, _st = report.summary_rows(self.res, self.cfg)
        price = dict(zip(_h, rows[0]))["Озвучил цену"]
        self.assertNotIn("re:", price)
        self.assertNotIn("\\b", price)

    def test_not_found_cell(self):
        _h, rows, _st = report.summary_rows(self.res, self.cfg)
        self.assertIn("не найдено", dict(zip(_h, rows[0]))["Узнал имя клиента"])

    def test_findings_are_time_ordered(self):
        header, rows = report.findings_rows(self.res)
        secs = [r[header.index("Секунда")] for r in rows]
        self.assertEqual(secs, sorted(secs))
        self.assertGreater(len(rows), 10)

    def test_findings_have_phrase_and_second(self):
        header, rows = report.findings_rows(self.res)
        i_phrase = header.index("Найдено в записи")
        i_sec = header.index("Секунда")
        checks = [r for r in rows if r[header.index("Тип")] == "пункт чек-листа"]
        self.assertTrue(checks)
        for r in checks:
            self.assertTrue(str(r[i_phrase]).strip(), "нет найденной фразы")
            self.assertIsInstance(r[i_sec], (int, float))

    def test_verdict_sheet(self):
        header, rows = report.verdict_rows(self.res, self.cfg)
        row = dict(zip(header, rows[0]))
        self.assertIn("Выполнено пунктов", row["Заключение"])
        self.assertTrue(row["Выполнено"])
        self.assertTrue(row["Что исправить"])

    def test_xlsx_has_all_sheets(self):
        from openpyxl import load_workbook

        with tempfile.TemporaryDirectory() as d:
            p = report.write_xlsx(self.res, self.cfg, os.path.join(d, "t.xlsx"))
            wb = load_workbook(p)
            for name in ("Итог", "Свод", "Находки", "Детали", "Товары"):
                self.assertIn(name, wb.sheetnames)

    def test_write_all(self):
        with tempfile.TemporaryDirectory() as d:
            written = report.write_all(self.res, self.cfg, d, "test")
            for fmt in ("xlsx", "csv", "json"):
                p = written.get(fmt, "")
                self.assertTrue(os.path.exists(p), f"{fmt}: {p}")
                self.assertGreater(os.path.getsize(p), 500)


class TestConfig(unittest.TestCase):
    def test_load_default(self):
        cfg = Config.load()
        self.assertGreater(len(cfg.checks), 5)
        self.assertGreater(len(cfg.products), 2)

    def test_roundtrip(self):
        cfg = Config.load()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.yaml")
            cfg.save(p)
            again = Config.load(p)
            self.assertEqual(len(cfg.checks), len(again.checks))

    def test_duplicate_id_rejected(self):
        cfg = Config.load()
        data = cfg.to_dict()
        data["checks"].append(data["checks"][0])
        with self.assertRaises(ValueError):
            Config.from_dict(data)

    def test_empty_check_does_not_block_saving(self):
        """Добавленный в интерфейсе пустой пункт не должен ломать сохранение."""
        data = Config.load().to_dict()
        data["checks"].append({"id": "new_one", "title": "Новый пункт",
                               "type": "keyword", "speaker": "operator", "weight": 5})
        cfg = Config.from_dict(data)          # не падает
        self.assertTrue(any("Новый пункт" in w for w in cfg.warnings))

    def test_warnings_not_written_to_yaml(self):
        cfg = Config.load()
        self.assertNotIn("warnings", cfg.to_dict())

    def test_duplicate_product_rejected(self):
        data = Config.load().to_dict()
        data["products"].append(dict(data["products"][0]))
        with self.assertRaises(ValueError):
            Config.from_dict(data)


class TestEmptyCheck(unittest.TestCase):
    """Пустой пункт обязан давать «н/д», а не молча засчитываться как «было»."""

    def setUp(self):
        data = Config.load().to_dict()
        data["checks"] = [{"id": "blank", "title": "Пустой", "type": "keyword",
                           "speaker": "operator", "weight": 5}]
        data["products"] = []
        self.cfg = Config.from_dict(data)
        self.cfg.llm.enabled = False

    def test_result_is_na(self):
        res = analyze_file("x.wav", self.cfg, transcript_override=demo_transcript())
        self.assertTrue(res.ok, res.error)
        self.assertIsNone(res.checks[0].passed)
        self.assertEqual(res.checks[0].status_ru, "н/д")

    def test_does_not_affect_score(self):
        res = analyze_file("x.wav", self.cfg, transcript_override=demo_transcript())
        self.assertEqual(res.score_max, 0.0, "пункт «н/д» не должен попадать в базу балла")


if __name__ == "__main__":
    unittest.main(verbosity=2)
