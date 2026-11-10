"""Тесты обучения правил на размеченных записях.

Самое важное здесь — не то, что движок находит закономерности,
а то, что он НЕ находит их там, где их нет. Поэтому половина тестов
подаёт заведомо случайные данные и требует пустого результата.
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from callqa import learn, stats  # noqa: E402
from callqa.config import Config  # noqa: E402
from callqa.metrics import compute as compute_metrics  # noqa: E402
from callqa.models import CLIENT, OPERATOR, Segment, Transcript, Word  # noqa: E402

CFG = Config.load()


def seg(spk: str, start: float, text: str, wps: float = 2.6) -> Segment:
    step = 1.0 / wps
    words, t = [], start
    for w in text.split():
        words.append(Word(w, round(t, 2), round(t + step, 2)))
        t += step
    return Segment(spk, start, round(t, 2), text, words)


def make_call(label: str, operator_lines, client_lines=(), name="c.wav",
              gap: float = 6.0) -> learn.LabeledCall:
    segs, t = [], 0.0
    for i in range(max(len(operator_lines), len(client_lines))):
        if i < len(operator_lines):
            segs.append(seg(OPERATOR, t, operator_lines[i]))
            t += gap
        if i < len(client_lines):
            segs.append(seg(CLIENT, t, client_lines[i]))
            t += gap
    tr = Transcript(segments=segs, duration=t + 2, channels=2, source="stereo")
    call = learn.LabeledCall(path=name, label=label, transcript=tr)
    call.metrics = compute_metrics(tr, CFG)
    return call


FILLER = ["Добрый день компания Светотехника чем могу помочь",
          "Сейчас посмотрю по базе наличие на складе",
          "Хорошо записал вашу заявку в систему"]
CLIENT_FILLER = ["Здравствуйте мне нужна лампа для работы",
                 "Понятно спасибо за информацию",
                 "Хорошо я вас понял"]


def corpus_with_signal(n_good: int = 14, n_bad: int = 14, seed: int = 0):
    """Успешные звонки содержат «расширенная гарантия», неуспешные — «я не знаю»."""
    rnd = random.Random(seed)
    calls = []
    for i in range(n_good):
        lines = list(FILLER)
        lines.insert(1, "Рекомендую оформить расширенную гарантию на два года")
        rnd.shuffle(lines)
        calls.append(make_call(learn.GOOD, lines, CLIENT_FILLER, f"g{i}.wav"))
    for i in range(n_bad):
        lines = list(FILLER)
        lines.insert(1, "Честно говоря я не знаю уточните позже")
        rnd.shuffle(lines)
        calls.append(make_call(learn.BAD, lines, CLIENT_FILLER, f"b{i}.wav"))
    return calls


def corpus_pure_noise(n: int = 16, seed: int = 7):
    """Метки расставлены случайно — настоящих отличий нет."""
    rnd = random.Random(seed)
    vocab = ["доставка", "гарантия", "оплата", "скидка", "менеджер", "склад",
             "заявка", "клиент", "карта", "срок", "цена", "модель"]
    calls = []
    for i in range(n * 2):
        lines = [" ".join(rnd.sample(vocab, 5)) + " для вас сегодня",
                 " ".join(rnd.sample(vocab, 5)) + " уточните пожалуйста",
                 " ".join(rnd.sample(vocab, 5)) + " по вашему заказу"]
        label = learn.GOOD if i % 2 == 0 else learn.BAD
        calls.append(make_call(label, lines, CLIENT_FILLER, f"n{i}.wav"))
    return calls


# --------------------------------------------------------------------------- #
class TestStatsGuards(unittest.TestCase):
    def test_fisher_no_association(self):
        self.assertAlmostEqual(stats.fisher_exact(5, 5, 5, 5), 1.0, places=6)

    def test_fisher_perfect_split_is_significant(self):
        self.assertLess(stats.fisher_exact(12, 0, 0, 12), 0.001)

    def test_fisher_tiny_sample_is_not_significant(self):
        """2 против 0 — красиво выглядит, но ничего не доказывает."""
        self.assertGreater(stats.fisher_exact(2, 0, 0, 2), 0.05)

    def test_fisher_symmetric(self):
        self.assertAlmostEqual(stats.fisher_exact(7, 2, 3, 8),
                               stats.fisher_exact(2, 7, 8, 3), places=12)

    def test_bh_is_monotone_and_bounded(self):
        p = [0.001, 0.01, 0.02, 0.2, 0.9]
        q = stats.benjamini_hochberg(p)
        self.assertTrue(all(0 <= x <= 1 for x in q))
        self.assertEqual(q, sorted(q))
        self.assertTrue(all(qi >= pi for qi, pi in zip(q, p)))

    def test_bh_raises_random_pvalues_above_threshold(self):
        """Тысяча случайных p: поправка должна почти всё забраковать."""
        rnd = random.Random(3)
        p = [rnd.random() for _ in range(1000)]
        q = stats.benjamini_hochberg(p)
        self.assertLess(sum(1 for x in q if x <= 0.1), 5)

    def test_auc_bounds(self):
        self.assertEqual(stats.auc([1, 2, 3], [1, 2, 3]), 0.5)
        self.assertEqual(stats.auc([5, 6], [1, 2]), 1.0)
        self.assertEqual(stats.auc([1, 2], [5, 6]), 0.0)


class TestNgrams(unittest.TestCase):
    def setUp(self):
        self.call = make_call(learn.GOOD, ["Рекомендую оформить расширенную гарантию"])

    def test_extracts_phrases(self):
        grams = learn.call_ngrams(self.call.transcript, OPERATOR)
        self.assertIn("рекомендовать", grams)
        # леммы даёт pymorphy, поэтому проверяем по исходному виду
        surfaces = {v[0].lower() for v in grams.values()}
        self.assertIn("расширенную гарантию", surfaces)
        self.assertIn("рекомендую оформить расширенную гарантию", surfaces)

    def test_ngram_never_spans_two_replies(self):
        """Фраза не должна склеивать конец одной реплики с началом следующей."""
        call = make_call(learn.GOOD, ["чем могу помочь", "честно говоря не знаю"])
        grams = learn.call_ngrams(call.transcript, OPERATOR)
        for surface, _pos in grams.values():
            self.assertNotIn("помочь честно", surface.lower())

    def test_counts_each_phrase_once_per_call(self):
        call = make_call(learn.GOOD, ["гарантия гарантия гарантия гарантия"])
        grams = learn.call_ngrams(call.transcript, OPERATOR)
        self.assertEqual(sum(1 for k in grams if k == "гарантия"), 1)

    def test_skips_pure_stopwords(self):
        call = make_call(learn.GOOD, ["ну вот и так это же"])
        self.assertEqual(learn.call_ngrams(call.transcript, OPERATOR), {})

    def test_position_is_relative(self):
        call = make_call(learn.GOOD, ["первая реплика оператора здесь",
                                      "вторая реплика оператора тут",
                                      "последняя реплика финальная"])
        grams = learn.call_ngrams(call.transcript, OPERATOR)
        self.assertLess(grams["первый"][1], grams["последний"][1])

    def test_speakers_are_separate(self):
        call = make_call(learn.GOOD, ["оператор говорит гарантия"],
                         ["клиент говорит доставка"])
        op = learn.call_ngrams(call.transcript, OPERATOR)
        cl = learn.call_ngrams(call.transcript, CLIENT)
        self.assertIn("гарантия", op)
        self.assertNotIn("гарантия", cl)
        self.assertIn("доставка", cl)


class TestMiningFindsRealSignal(unittest.TestCase):
    def setUp(self):
        self.calls = corpus_with_signal()
        self.found = learn.mine_phrases(self.calls)
        self.texts = {c.phrase for c in self.found}

    def test_finds_positive_marker(self):
        self.assertTrue(any("гарант" in t for t in self.texts),
                        f"не нашлась заложенная фраза, найдено: {self.texts}")

    def test_finds_negative_marker(self):
        neg = [c for c in self.found if c.direction == learn.BAD]
        self.assertTrue(any("знаю" in c.phrase for c in neg),
                        f"не нашлась фраза неудачи: {[c.phrase for c in neg]}")

    def test_directions_are_correct(self):
        for c in self.found:
            if "гарант" in c.phrase:
                self.assertEqual(c.direction, learn.GOOD)
            if "знаю" in c.phrase:
                self.assertEqual(c.direction, learn.BAD)

    def test_common_filler_is_not_reported(self):
        """Фразы, встречающиеся в обеих группах, правилами быть не должны."""
        for c in self.found:
            self.assertNotIn("чем могу помочь", c.phrase)
            self.assertNotIn("посмотрю по базе", c.phrase)

    def test_q_values_are_small(self):
        for c in self.found:
            self.assertLessEqual(c.q_value, 0.10)

    def test_sliding_windows_are_collapsed(self):
        """Одно предложение не должно превратиться в десяток похожих правил."""
        self.assertLessEqual(len(self.found), 4,
                             f"слишком много почти одинаковых правил: "
                             f"{[c.phrase for c in self.found]}")

    def test_variants_are_preserved(self):
        """Схлопнутые формулировки должны остаться видимыми человеку."""
        self.assertTrue(any(c.variants for c in self.found),
                        "варианты потерялись — непонятно, из чего выбирали")

    def test_representative_is_not_a_fragment(self):
        """Фраза-правило не должна обрываться на прилагательном или предлоге."""
        for c in self.found:
            self.assertFalse(learn._dangling(c.phrase.split()),
                             f"обрывок вместо фразы: «{c.phrase}»")


class TestMiningRejectsNoise(unittest.TestCase):
    """Самое важное: на случайной разметке правил быть не должно."""

    def test_no_rules_from_random_labels(self):
        found = learn.mine_phrases(corpus_pure_noise())
        self.assertEqual(found, [], f"выдуманные правила: {[c.phrase for c in found]}")

    def test_no_rules_from_identical_corpora(self):
        calls = [make_call(learn.GOOD, FILLER, CLIENT_FILLER, f"g{i}.wav")
                 for i in range(10)]
        calls += [make_call(learn.BAD, FILLER, CLIENT_FILLER, f"b{i}.wav")
                  for i in range(10)]
        self.assertEqual(learn.mine_phrases(calls), [])

    def test_single_occurrence_is_ignored(self):
        calls = [make_call(learn.GOOD, FILLER + ["уникальная случайная реплика"],
                           CLIENT_FILLER, "g0.wav")]
        calls += [make_call(learn.GOOD, FILLER, CLIENT_FILLER, f"g{i}.wav")
                  for i in range(1, 10)]
        calls += [make_call(learn.BAD, FILLER, CLIENT_FILLER, f"b{i}.wav")
                  for i in range(10)]
        found = learn.mine_phrases(calls, min_calls=3)
        self.assertFalse(any("уникальн" in c.phrase for c in found))

    def test_empty_side_returns_nothing(self):
        calls = [make_call(learn.GOOD, FILLER, name=f"g{i}.wav") for i in range(5)]
        self.assertEqual(learn.mine_phrases(calls), [])


class TestOutcomeRisk(unittest.TestCase):
    """Фраза в самом конце успешного разговора — скорее следствие, чем причина."""

    def test_late_phrase_is_flagged(self):
        calls = []
        for i in range(12):
            calls.append(make_call(learn.GOOD, FILLER + ["оформляем ваш заказ прямо сейчас"],
                                   CLIENT_FILLER, f"g{i}.wav"))
            calls.append(make_call(learn.BAD, FILLER, CLIENT_FILLER, f"b{i}.wav"))
        found = learn.mine_phrases(calls)
        late = [c for c in found if "оформля" in c.phrase or "заказ" in c.phrase]
        self.assertTrue(late, "фраза не найдена вовсе")
        self.assertTrue(any(c.outcome_risk for c in late),
                        "поздняя фраза не помечена как вероятное следствие")

    def test_flagged_phrases_excluded_by_default(self):
        res = learn.LearnResult(phrases=[
            learn.PhraseCandidate(phrase="оформляем заказ", display="оформляем заказ",
                                  speaker=OPERATOR, direction=learn.GOOD,
                                  good_calls=10, bad_calls=1, good_total=12,
                                  bad_total=12, p_value=0.001, q_value=0.01,
                                  lift=8.0, mean_position=0.95, outcome_risk=True),
            learn.PhraseCandidate(phrase="какой бюджет", display="какой бюджет",
                                  speaker=OPERATOR, direction=learn.GOOD,
                                  good_calls=9, bad_calls=1, good_total=12,
                                  bad_total=12, p_value=0.002, q_value=0.01,
                                  lift=7.0, mean_position=0.2, outcome_risk=False),
        ])
        checks = learn.build_checks(res)
        titles = [c["title"] for c in checks]
        self.assertEqual(len(checks), 1)
        self.assertIn("какой бюджет", titles[0])

    def test_can_be_included_explicitly(self):
        res = learn.LearnResult(phrases=[
            learn.PhraseCandidate(phrase="оформляем заказ", display="оформляем заказ",
                                  speaker=OPERATOR, direction=learn.GOOD,
                                  good_calls=10, bad_calls=1, good_total=12,
                                  bad_total=12, p_value=0.001, q_value=0.01,
                                  lift=8.0, mean_position=0.95, outcome_risk=True)])
        self.assertEqual(len(learn.build_checks(res, phrase_ids=[0])), 1)


class TestChecksGeneration(unittest.TestCase):
    def setUp(self):
        self.res = learn.analyse("", "", CFG, calls=corpus_with_signal())

    def test_positive_and_negative_checks(self):
        checks = learn.build_checks(self.res)
        self.assertTrue(checks)
        self.assertTrue(any(not c["invert"] for c in checks))
        self.assertTrue(any(c["invert"] for c in checks),
                        "признак неудачи должен стать инвертированным пунктом")

    def test_checks_are_valid_config(self):
        checks = learn.build_checks(self.res)
        merged = learn.merge_into_config(CFG, checks)
        self.assertEqual(len(merged.checks), len(CFG.checks) + len(checks))

    def test_ids_are_unique(self):
        checks = learn.build_checks(self.res)
        ids = [c["id"] for c in checks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_existing_checks_untouched(self):
        """Обучение обязано добавлять, а не переписывать ручные настройки."""
        before = {c.id: c.title for c in CFG.checks}
        merged = learn.merge_into_config(CFG, learn.build_checks(self.res))
        after = {c.id: c.title for c in merged.checks}
        for cid, title in before.items():
            self.assertEqual(after.get(cid), title)

    def test_comment_explains_numbers(self):
        checks = learn.build_checks(self.res)
        self.assertTrue(all("Обучение:" in c["comment"] for c in checks))

    def test_generated_checks_actually_work(self):
        """Сгенерированный пункт должен срабатывать на исходной записи."""
        from callqa.pipeline import analyze_file

        checks = learn.build_checks(self.res)
        positive = next(c for c in checks if not c["invert"])
        data = CFG.to_dict()
        data["checks"] = [positive]
        data["products"] = []
        cfg = Config.from_dict(data)
        cfg.llm.enabled = False
        cfg.filters.min_words = 1
        cfg.filters.min_duration_sec = 0
        good = next(c for c in corpus_with_signal() if c.label == learn.GOOD)
        res = analyze_file("x.wav", cfg, transcript_override=good.transcript)
        self.assertTrue(res.checks[0].passed, res.checks[0].detail)


class TestValidation(unittest.TestCase):
    def test_real_signal_validates_well(self):
        res = learn.analyse("", "", CFG, calls=corpus_with_signal(20, 20))
        v = res.validation
        self.assertTrue(v.get("ok"), v)
        self.assertGreater(v["accuracy"], 0.8,
                           "на явном сигнале точность должна быть высокой")

    def test_noise_is_reported_as_useless(self):
        res = learn.analyse("", "", CFG, calls=corpus_pure_noise(14))
        self.assertEqual(res.phrases, [])
        self.assertTrue(any("не найдено" in w for w in res.warnings))

    def test_small_sample_warning(self):
        res = learn.analyse("", "", CFG, calls=corpus_with_signal(3, 3))
        self.assertTrue(any("мало" in w for w in res.warnings))

    def test_imbalance_warning(self):
        res = learn.analyse("", "", CFG, calls=corpus_with_signal(20, 4))
        self.assertTrue(any("разного размера" in w for w in res.warnings))

    def test_duration_confound_warning(self):
        calls = []
        for i in range(12):
            calls.append(make_call(learn.GOOD, FILLER * 4, CLIENT_FILLER, f"g{i}.wav"))
            calls.append(make_call(learn.BAD, FILLER[:1], CLIENT_FILLER[:1], f"b{i}.wav"))
        res = learn.analyse("", "", CFG, calls=calls)
        self.assertTrue(any("длительност" in w for w in res.warnings),
                        f"перекос по длине не замечен: {res.warnings}")

    def test_missing_folder_warning(self):
        calls = [make_call(learn.GOOD, FILLER, name=f"g{i}.wav") for i in range(5)]
        res = learn.analyse("", "", CFG, calls=calls)
        self.assertTrue(any("обеих папках" in w for w in res.warnings))


class TestMetricMining(unittest.TestCase):
    def test_finds_separating_metric(self):
        calls = []
        for i in range(15):                       # оператор не перебивает
            calls.append(make_call(learn.GOOD, FILLER, CLIENT_FILLER, f"g{i}.wav"))
        for i in range(15):                       # оператор постоянно перебивает
            segs = [seg(CLIENT, 0.0, "Мне нужна лампа для работы за компьютером вечером"),
                    seg(OPERATOR, 2.0, "Так вот смотрите у нас есть отличная модель"),
                    seg(CLIENT, 9.0, "А сколько она стоит и какая там гарантия"),
                    seg(OPERATOR, 11.0, "Значит слушайте гарантия год всё понятно")]
            tr = Transcript(segments=segs, duration=25, channels=2, source="stereo")
            c = learn.LabeledCall(path=f"b{i}.wav", label=learn.BAD, transcript=tr)
            c.metrics = compute_metrics(tr, CFG)
            calls.append(c)
        found = learn.mine_metrics(calls)
        self.assertTrue(any("interrupt" in m.metric for m in found),
                        f"перебивания не найдены: {[m.metric for m in found]}")

    def test_ignores_practically_meaningless_difference(self):
        """Разница «156.0 против 156.1» разделяет идеально, но смысла не имеет."""
        calls = []
        for i in range(12):
            c = make_call(learn.GOOD, FILLER, CLIENT_FILLER, f"g{i}.wav")
            c.metrics = {"operator_wpm": 156.0 + i * 0.001,
                         "operator_interruptions": 0}
            calls.append(c)
        for i in range(12):
            c = make_call(learn.BAD, FILLER, CLIENT_FILLER, f"b{i}.wav")
            c.metrics = {"operator_wpm": 156.1 + i * 0.001,
                         "operator_interruptions": 0}
            calls.append(c)
        found = learn.mine_metrics(calls)
        self.assertEqual([m.metric for m in found], [],
                         "мизерная разница не должна становиться правилом")

    def test_keeps_meaningful_difference(self):
        calls = []
        for i in range(12):
            c = make_call(learn.GOOD, FILLER, CLIENT_FILLER, f"g{i}.wav")
            c.metrics = {"operator_interruptions": 0 + (i % 2)}
            calls.append(c)
        for i in range(12):
            c = make_call(learn.BAD, FILLER, CLIENT_FILLER, f"b{i}.wav")
            c.metrics = {"operator_interruptions": 5 + (i % 2)}
            calls.append(c)
        found = learn.mine_metrics(calls)
        self.assertTrue(any(m.metric == "operator_interruptions" for m in found))

    def test_no_metrics_from_identical_groups(self):
        calls = [make_call(learn.GOOD, FILLER, CLIENT_FILLER, f"g{i}.wav")
                 for i in range(12)]
        calls += [make_call(learn.BAD, FILLER, CLIENT_FILLER, f"b{i}.wav")
                  for i in range(12)]
        self.assertEqual(learn.mine_metrics(calls), [])


class TestCache(unittest.TestCase):
    def test_roundtrip_preserves_transcript(self):
        call = make_call(learn.GOOD, ["Рекомендую расширенную гарантию"],
                         ["Сколько стоит"])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            learn._save_cache(p, call.transcript, call.metrics)
            tr, m = learn._load_cache(p)
        self.assertEqual(len(tr.segments), len(call.transcript.segments))
        self.assertEqual(tr.segments[0].text, call.transcript.segments[0].text)
        self.assertEqual(tr.segments[0].speaker, OPERATOR)
        self.assertTrue(tr.segments[0].words)
        self.assertIn("operator_talk_ratio", m)

    def test_missing_cache_returns_none(self):
        self.assertIsNone(learn._load_cache("/нет/такого/файла.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
