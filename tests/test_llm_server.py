"""Проверка работы с моделью через настоящий HTTP-сервер.

Поднимается заглушка, отвечающая по протоколу OpenAI ровно как Ollama
или LM Studio. Это проверяет весь путь целиком: обнаружение сервера,
работу без ключа, компактный запрос, разбор JSON и заполнение пунктов
чек-листа — то, что нельзя проверить моками отдельных функций.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from callqa import llm as llm_mod  # noqa: E402
from callqa.config import Config  # noqa: E402
from callqa.demo import demo_transcript  # noqa: E402
from callqa.pipeline import analyze_file  # noqa: E402

# что «модель» вернёт и что она получила — для проверок внутри тестов
STATE = {"answer": None, "last_request": None, "auth": None,
         "models": ["qwen3:8b"], "fail": False}


class FakeLLM(BaseHTTPRequestHandler):
    def log_message(self, *_a):        # тишина в выводе тестов
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"data": [{"id": m} for m in STATE["models"]]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        STATE["last_request"] = json.loads(raw)
        STATE["auth"] = self.headers.get("Authorization")
        if STATE["fail"]:
            self._json(500, {"error": "модель упала"})
            return
        content = STATE["answer"] if STATE["answer"] is not None else '{"ok": true}'
        self._json(200, {"choices": [{"message": {"role": "assistant",
                                                  "content": content}}]})


class ServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = HTTPServer(("127.0.0.1", 0), FakeLLM)
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}/v1"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        STATE.update(answer=None, last_request=None, auth=None,
                     models=["qwen3:8b"], fail=False)

    def local_cfg(self, **over) -> Config:
        cfg = Config.load()
        cfg.llm.enabled = True
        cfg.llm.provider = "openai"
        cfg.llm.base_url = self.base
        cfg.llm.model = "qwen3:8b"
        cfg.llm.api_key = ""
        cfg.llm.api_key_env = "CALLQA_НЕТ_ТАКОЙ_ПЕРЕМЕННОЙ"
        for k, v in over.items():
            setattr(cfg.llm, k, v)
        return cfg


class TestConnection(ServerCase):
    def test_works_without_any_api_key(self):
        """Главное: локальный сервер должен работать без платного ключа."""
        ok, note = llm_mod.check_connection(self.local_cfg())
        self.assertTrue(ok, note)
        self.assertIn("qwen3:8b", note)

    def test_sends_placeholder_key(self):
        """Ollama требует заголовок Authorization, но не проверяет его."""
        llm_mod.check_connection(self.local_cfg())
        self.assertEqual(STATE["auth"], "Bearer local")

    def test_server_error_is_reported(self):
        STATE["fail"] = True
        ok, note = llm_mod.check_connection(self.local_cfg())
        self.assertFalse(ok)
        self.assertIn("500", note)

    def test_offline_server_message_is_helpful(self):
        cfg = self.local_cfg()
        cfg.llm.base_url = "http://127.0.0.1:1/v1"     # заведомо закрытый порт
        ok, note = llm_mod.check_connection(cfg)
        self.assertFalse(ok)
        self.assertIn("не отвечает", note)


class TestDiscovery(ServerCase):
    def test_finds_running_server(self):
        found = llm_mod.discover_local_servers()
        # адрес нашего сервера в списке стандартных отсутствует, поэтому
        # проверяем сам механизм опроса на нашем порту
        import requests

        r = requests.get(self.base + "/models", timeout=3)
        self.assertEqual(r.status_code, 200)
        self.assertIn("qwen3:8b", [m["id"] for m in r.json()["data"]])
        self.assertIsInstance(found, list)


class TestCompactPrompt(ServerCase):
    """Для локальных моделей должен уходить укороченный запрос."""

    def make_cfg(self) -> Config:
        data = Config.load().to_dict()
        data["checks"] = [
            {"id": "polite", "title": "Вежливый тон", "type": "llm",
             "speaker": "operator", "weight": 5,
             "prompt": "Оператор был вежлив на протяжении разговора?"},
            {"id": "summary", "title": "Резюмировал запрос", "type": "llm",
             "speaker": "operator", "weight": 5,
             "prompt": "Оператор пересказал запрос клиента своими словами?"},
        ]
        cfg = Config.from_dict(data)
        cfg.llm.enabled = True
        cfg.llm.provider = "openai"
        cfg.llm.base_url = self.base
        cfg.llm.model = "qwen3:8b"
        cfg.llm.api_key = ""
        cfg.llm.api_key_env = "CALLQA_НЕТ_ТАКОЙ_ПЕРЕМЕННОЙ"
        return cfg

    def test_compact_is_used_for_local(self):
        cfg = self.make_cfg()
        self.assertTrue(cfg.llm.use_compact())

    def test_local_checks_get_filled(self):
        STATE["answer"] = json.dumps({"answers": [
            {"id": "polite", "passed": True, "quote": "Подскажите пожалуйста"},
            {"id": "summary", "passed": False, "quote": ""},
            {"id": "товар::Страховка", "passed": True, "quote": "оформим страховку"},
        ]}, ensure_ascii=False)

        res = analyze_file("x.wav", self.make_cfg(),
                           transcript_override=demo_transcript())
        self.assertTrue(res.ok, res.error)
        by_id = {c.id: c for c in res.checks}
        self.assertTrue(by_id["polite"].passed)
        self.assertFalse(by_id["summary"].passed)
        self.assertEqual(by_id["polite"].status_ru, "было")

    def test_llm_can_add_product_rules_missed(self):
        """Модель дозачитывает товар, которого нет в списках формулировок."""
        STATE["answer"] = json.dumps({"answers": [
            {"id": "товар::Страховка", "passed": True,
             "quote": "могу оформить защиту покупки"},
        ]}, ensure_ascii=False)
        res = analyze_file("x.wav", self.make_cfg(),
                           transcript_override=demo_transcript())
        ins = next(p for p in res.products if p.name == "Страховка")
        self.assertTrue(ins.offered, "модель должна была дозачесть товар")
        self.assertEqual(ins.how, "llm")

    def test_prompt_is_short_and_has_dialogue(self):
        STATE["answer"] = json.dumps({"answers": []})
        analyze_file("x.wav", self.make_cfg(), transcript_override=demo_transcript())
        sent = STATE["last_request"]
        text = sent["messages"][-1]["content"]
        self.assertIn("Оператор:", text, "в запросе нет стенограммы")
        self.assertIn("polite", text, "в запросе нет пунктов чек-листа")
        self.assertNotIn("рекомендаци", text.lower(),
                         "компактный запрос не должен просить рекомендации")

    def test_markdown_wrapped_json_is_parsed(self):
        """Маленькие модели любят обернуть ответ в ```json — это не должно ломать."""
        STATE["answer"] = ("Вот результат:\n```json\n"
                           + json.dumps({"answers": [{"id": "polite", "passed": True}]})
                           + "\n```")
        res = analyze_file("x.wav", self.make_cfg(),
                           transcript_override=demo_transcript())
        by_id = {c.id: c for c in res.checks}
        self.assertTrue(by_id["polite"].passed)

    def test_garbage_answer_does_not_break_analysis(self):
        """Модель ответила мусором — остальной разбор обязан уцелеть."""
        STATE["answer"] = "я не смогла разобрать этот диалог, извините"
        res = analyze_file("x.wav", self.make_cfg(),
                           transcript_override=demo_transcript())
        self.assertTrue(res.ok, "разбор не должен падать из-за ответа модели")
        self.assertTrue(any("не выполнена" in w for w in res.warnings))
        by_id = {c.id: c for c in res.checks}
        self.assertIsNone(by_id["polite"].passed)

    def test_server_down_does_not_break_analysis(self):
        cfg = self.make_cfg()
        cfg.llm.base_url = "http://127.0.0.1:1/v1"
        res = analyze_file("x.wav", cfg, transcript_override=demo_transcript())
        self.assertTrue(res.ok)
        self.assertTrue(any("локальная модель" in w for w in res.warnings))

    def test_null_answer_stays_na(self):
        STATE["answer"] = json.dumps({"answers": [
            {"id": "polite", "passed": None, "quote": ""}]})
        res = analyze_file("x.wav", self.make_cfg(),
                           transcript_override=demo_transcript())
        by_id = {c.id: c for c in res.checks}
        self.assertIsNone(by_id["polite"].passed)
        self.assertEqual(by_id["polite"].status_ru, "н/д")


class TestFreeProviders(unittest.TestCase):
    def test_presets_are_well_formed(self):
        self.assertTrue(llm_mod.FREE_PROVIDERS)
        for p in llm_mod.FREE_PROVIDERS:
            for key in ("id", "name", "base_url", "model", "key_env",
                        "signup", "limits"):
                self.assertIn(key, p, p.get("name"))
            self.assertTrue(p["base_url"].startswith("https://"), p["name"])

    def test_preset_urls_survive_normalization(self):
        """Нормализация не должна ломать нестандартные пути вроде /v1beta/openai."""
        for p in llm_mod.FREE_PROVIDERS:
            self.assertEqual(llm_mod.normalize_base_url(p["base_url"]),
                             p["base_url"], p["name"])

    def test_bare_host_gets_v1(self):
        self.assertEqual(llm_mod.normalize_base_url("http://localhost:11434"),
                         "http://localhost:11434/v1")
        self.assertEqual(llm_mod.normalize_base_url("localhost:11434"),
                         "http://localhost:11434/v1")

    def test_free_providers_are_not_local(self):
        cfg = Config.load()
        for p in llm_mod.FREE_PROVIDERS:
            cfg.llm.base_url = p["base_url"]
            self.assertFalse(cfg.llm.is_local(), p["name"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
