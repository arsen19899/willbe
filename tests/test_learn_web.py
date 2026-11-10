"""Проверка обучения через веб-интерфейс — от папок с файлами до пунктов чек-листа.

Распознавание речи не запускается: расшифровки заранее кладутся в кэш,
тот самый, которым пользуется обучение при повторных прогонах.
Это заодно проверяет, что кэш действительно работает.
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
import time
import unittest
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from callqa import learn  # noqa: E402
from callqa.config import DEFAULT_CONFIG_PATH, Config  # noqa: E402
from callqa.metrics import compute as compute_metrics  # noqa: E402
from callqa.models import CLIENT, OPERATOR, Segment, Transcript, Word  # noqa: E402
from callqa.web import app as webapp  # noqa: E402

FILLER = ["Добрый день компания Светотехника чем могу помочь",
          "Сейчас посмотрю по базе наличие на складе",
          "Хорошо записал вашу заявку в систему"]
CLIENT_LINES = ["Здравствуйте мне нужна лампа для работы",
                "Понятно спасибо за информацию",
                "Хорошо я вас понял"]


def _seg(spk, start, text, wps=2.6):
    step, words, t = 1 / wps, [], start
    for w in text.split():
        words.append(Word(w, round(t, 2), round(t + step, 2)))
        t += step
    return Segment(spk, start, round(t, 2), text, words)


def _transcript(op_lines, cl_lines):
    segs, t = [], 0.0
    for i in range(max(len(op_lines), len(cl_lines))):
        if i < len(op_lines):
            segs.append(_seg(OPERATOR, t, op_lines[i]))
            t += 6
        if i < len(cl_lines):
            segs.append(_seg(CLIENT, t, cl_lines[i]))
            t += 6
    return Transcript(segments=segs, duration=t + 2, channels=2, source="stereo")


def _silent_wav(path: str, seconds: float, seed: int) -> None:
    """Файл нужен только чтобы существовать: содержимое влияет лишь на ключ кэша."""
    rnd = random.Random(seed)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(bytes(rnd.randrange(256) for _ in range(int(8000 * seconds) * 2)))


class LearnWebCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.good = os.path.join(cls.tmp, "успешные")
        cls.bad = os.path.join(cls.tmp, "неуспешные")
        os.makedirs(cls.good)
        os.makedirs(cls.bad)
        cls.cache = os.path.join(cls.tmp, "reports", learn.CACHE_DIR)

        cfg = Config.load()
        cfg.output.dir = os.path.join(cls.tmp, "reports")
        cls.cfg_path = os.path.join(cls.tmp, "cfg.yaml")
        cfg.save(cls.cfg_path)

        rnd = random.Random(0)
        for i in range(14):
            lines = list(FILLER)
            lines.insert(1, "Рекомендую оформить расширенную гарантию на два года")
            rnd.shuffle(lines)
            cls._make(os.path.join(cls.good, f"g{i}.wav"), lines, i, cfg)
        for i in range(14):
            lines = list(FILLER)
            lines.insert(1, "Честно говоря я не знаю уточните позже")
            rnd.shuffle(lines)
            cls._make(os.path.join(cls.bad, f"b{i}.wav"), lines, 100 + i, cfg)

        webapp.STATE["config_path"] = cls.cfg_path
        cls.client = TestClient(webapp.app)

    @classmethod
    def _make(cls, path, op_lines, seed, cfg):
        _silent_wav(path, 12, seed)
        tr = _transcript(op_lines, CLIENT_LINES)
        m = compute_metrics(tr, cfg)
        learn._save_cache(learn._cache_path(cls.cache, path, cfg.asr.model), tr, m)

    @classmethod
    def tearDownClass(cls):
        import shutil

        webapp.STATE["config_path"] = DEFAULT_CONFIG_PATH
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_learn(self, **body) -> dict:
        payload = {"good": self.good, "bad": self.bad}
        payload.update(body)
        r = self.client.post("/api/learn", json=payload)
        self.assertEqual(r.status_code, 200, r.text)
        job = r.json()["job_id"]
        for _ in range(200):
            s = self.client.get(f"/api/learn/{job}").json()
            if s["status"] in ("done", "error"):
                break
            time.sleep(0.05)
        self.assertEqual(s["status"], "done", s.get("error"))
        return job, s["result"]


class TestLearnEndpoint(LearnWebCase):
    def test_finds_signal_through_api(self):
        _job, res = self.run_learn()
        self.assertEqual(res["good_total"], 14)
        self.assertEqual(res["bad_total"], 14)
        phrases = [p["display"] for p in res["phrases"]]
        good = [p["display"] for p in res["phrases"] if p["direction"] == "good"]
        bad = [p["display"] for p in res["phrases"] if p["direction"] == "bad"]
        self.assertTrue(any("гарант" in p or "оформить" in p for p in good), phrases)
        self.assertTrue(any("знаю" in p for p in bad), phrases)

    def test_no_near_duplicate_rules(self):
        """Скользящие окна по одному предложению схлопнуты в одно правило."""
        _job, res = self.run_learn()
        self.assertLessEqual(len(res["phrases"]), 4,
                             [p["display"] for p in res["phrases"]])

    def test_used_cache_not_asr(self):
        """Если бы кэш не сработал, распознавание бы упало — моделей нет."""
        _job, res = self.run_learn()
        self.assertEqual(res["failed"], [])

    def test_validation_reported(self):
        _job, res = self.run_learn()
        self.assertTrue(res["validation"]["ok"], res["validation"])
        self.assertGreater(res["validation"]["accuracy"], 0.8)

    def test_bad_folder_rejected(self):
        r = self.client.post("/api/learn",
                             json={"good": "/нет/такой", "bad": self.bad})
        self.assertEqual(r.status_code, 400)
        self.assertIn("успешных", r.json()["detail"])

    def test_stricter_settings_yield_fewer(self):
        _j1, loose = self.run_learn(max_q=0.2, min_lift=1.2)
        _j2, strict = self.run_learn(max_q=0.01, min_lift=3.0)
        self.assertLessEqual(len(strict["phrases"]), len(loose["phrases"]))


class TestApply(LearnWebCase):
    def test_preview_returns_yaml(self):
        job, res = self.run_learn()
        r = self.client.post(f"/api/learn/{job}/preview",
                             json={"phrases": [0], "metrics": []})
        self.assertEqual(r.status_code, 200)
        self.assertIn("checks:", r.json()["yaml"])
        self.assertEqual(r.json()["count"], 1)

    def test_apply_adds_to_checklist(self):
        job, res = self.run_learn()
        before = len(self.client.get("/api/config").json()["config"]["checks"])
        r = self.client.post(f"/api/learn/{job}/apply",
                             json={"phrases": [0, 1], "metrics": []})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["added"], 2)
        after = self.client.get("/api/config").json()["config"]["checks"]
        self.assertEqual(len(after), before + 2)

    def test_apply_keeps_existing_checks(self):
        job, _res = self.run_learn()
        before = self.client.get("/api/config").json()["config"]["checks"]
        titles_before = {c["id"]: c["title"] for c in before}
        self.client.post(f"/api/learn/{job}/apply",
                         json={"phrases": [0], "metrics": []})
        after = self.client.get("/api/config").json()["config"]["checks"]
        for c in after:
            if c["id"] in titles_before:
                self.assertEqual(c["title"], titles_before[c["id"]],
                                 "ручной пункт изменился")

    def test_added_checks_are_editable(self):
        """Добавленный пункт должен нормально редактироваться и сохраняться."""
        job, _res = self.run_learn()
        self.client.post(f"/api/learn/{job}/apply",
                         json={"phrases": [0], "metrics": []})
        cfg = self.client.get("/api/config").json()["config"]
        learned = [c for c in cfg["checks"] if c["id"].startswith("learn_")]
        self.assertTrue(learned)
        learned[-1]["title"] = "Переименовано вручную"
        learned[-1]["weight"] = 9
        r = self.client.post("/api/config", json={"config": cfg})
        self.assertEqual(r.status_code, 200, r.text)
        again = self.client.get("/api/config").json()["config"]["checks"]
        found = [c for c in again if c["title"] == "Переименовано вручную"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["weight"], 9)

    def test_apply_nothing_selected(self):
        job, _res = self.run_learn()
        r = self.client.post(f"/api/learn/{job}/apply",
                             json={"phrases": [], "metrics": []})
        self.assertEqual(r.status_code, 400)

    def test_unknown_job(self):
        r = self.client.post("/api/learn/zzz/apply", json={"phrases": [0]})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
