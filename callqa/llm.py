"""Смысловая оценка диалога языковой моделью.

Одним запросом на звонок закрываются все пункты чек-листа типа ``llm``,
перепроверяются товары (перефразированные предложения правила не ловят)
и формируется краткое резюме с рекомендациями.
Работает через Anthropic API или любой OpenAI-совместимый шлюз.
Если ключа нет — пункты помечаются «н/д», остальной анализ не страдает.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .models import CheckResult, Evidence, ProductResult, Transcript

SYSTEM = (
    "Ты — методист отдела контроля качества колл-центра. Ты оцениваешь стенограмму "
    "телефонного разговора между оператором и клиентом строго по фактам из текста. "
    "Не додумывай то, чего в стенограмме нет. Если данных недостаточно для вывода, "
    "честно ставь null. Отвечай только валидным JSON без markdown-обёртки."
)


def _build_compact_prompt(transcript: Transcript, cfg: Config,
                          llm_checks: List, products: List[ProductResult]) -> str:
    """Урезанный запрос для небольших локальных моделей.

    Только вердикты «да/нет» — без резюме и рекомендаций. Это втрое сокращает
    ответ, а значит и время генерации, и заметно повышает шанс, что модель
    на 7–8 млрд параметров вернёт корректный JSON.
    """
    dialogue = transcript.dialogue(max_chars=24000)
    items = []
    for c in llm_checks:
        items.append(f'{{"id": "{c.id}", "вопрос": "{_one_line(c.prompt or c.title)}"}}')
    for p in products:
        items.append(f'{{"id": "товар::{p.name}", '
                     f'"вопрос": "Оператор предлагал клиенту: {p.name}?"}}')

    return f"""Стенограмма телефонного разговора:

{dialogue}

Ответь по каждому вопросу: да (true) или нет (false).
Отвечай только по фактам из стенограммы. Не уверен — ставь null.

Вопросы:
{chr(10).join('- ' + i for i in items)}

Верни ТОЛЬКО JSON, без пояснений и без markdown:
{{"answers": [{{"id": "...", "passed": true, "quote": "цитата или пустая строка"}}]}}"""


def _one_line(s: str) -> str:
    return " ".join((s or "").split()).replace('"', "'")


def _build_prompt(transcript: Transcript, cfg: Config,
                  llm_checks: List, products: List[ProductResult],
                  metrics: Dict[str, Any]) -> str:
    dialogue = transcript.dialogue(max_chars=60000)

    checks_block = "\n".join(
        f'- id: "{c.id}"\n  вопрос: {(c.prompt or c.title).strip()}'
        for c in llm_checks
    ) or "- (пунктов нет)"

    products_block = "\n".join(
        f'- "{p.name}" (варианты формулировок: {", ".join(a for a in _aliases(cfg, p.name))})'
        for p in products
    ) or "- (товаров нет)"

    facts = (
        f"Длительность: {metrics.get('duration')} с; "
        f"доля речи оператора: {metrics.get('operator_talk_ratio')}; "
        f"перебиваний оператором: {metrics.get('operator_interruptions')}; "
        f"перебиваний клиентом: {metrics.get('client_interruptions')}."
    )

    return f"""Ниже стенограмма разговора. Метрики, посчитанные автоматически: {facts}

=== СТЕНОГРАММА ===
{dialogue}
=== КОНЕЦ СТЕНОГРАММЫ ===

ЗАДАЧА 1. Ответь по каждому пункту чек-листа: выполнен он или нет.
Пункты:
{checks_block}

ЗАДАЧА 2. Определи, предлагал ли оператор клиенту каждый из товаров/услуг.
Засчитывай и перефразированные предложения, а не только точные слова.
Упоминание клиентом не считается предложением оператора.
Товары:
{products_block}

ЗАДАЧА 3. Дай краткое резюме разговора (2-3 предложения), оценку качества
работы оператора от 0 до 100 и до 5 конкретных рекомендаций.

Верни JSON строго такой структуры:
{{
  "checks": [
    {{"id": "...", "passed": true|false|null, "confidence": 0.0-1.0,
      "reason": "коротко, почему", "quote": "дословная цитата из стенограммы или пустая строка"}}
  ],
  "products": [
    {{"name": "...", "offered": true|false, "confidence": 0.0-1.0,
      "quote": "дословная цитата предложения или пустая строка"}}
  ],
  "summary": "...",
  "quality_score": 0-100,
  "client_sentiment": "положительный|нейтральный|отрицательный",
  "call_outcome": "сделка|договорённость о следующем шаге|отказ|консультация|не определено",
  "recommendations": ["...", "..."]
}}"""


def _aliases(cfg: Config, name: str) -> List[str]:
    for p in cfg.products:
        if p.name == name:
            return [p.name] + list(p.aliases or [])
    return [name]


# --------------------------------------------------------------------------- #
def evaluate(cfg: Config, transcript: Transcript, results: List[CheckResult],
             products: List[ProductResult], metrics: Dict[str, Any]
             ) -> Tuple[str, List[str], Dict[str, Any], List[str]]:
    """Дополняет results и products смысловой оценкой.
    Возвращает (резюме, рекомендации, доп. поля, предупреждения)."""
    warnings: List[str] = []
    llm_checks = [c for c in cfg.checks if c.type == "llm"]
    if not cfg.llm.enabled or cfg.llm.provider == "none":
        return "", [], {}, []
    if not llm_checks and not products:
        return "", [], {}, []

    key = cfg.llm_api_key()
    if not key:
        return "", [], {}, [
            "Смысловые пункты не проверены: не задан API-ключ "
            f"(переменная окружения {cfg.llm.api_key_env}). "
            "В отчёте они помечены «н/д» и на балл не влияют."
        ]

    compact = cfg.llm.use_compact()
    prompt = (_build_compact_prompt(transcript, cfg, llm_checks, products)
              if compact else
              _build_prompt(transcript, cfg, llm_checks, products, metrics))
    try:
        raw = _call(cfg, prompt, key)
        data = _parse_json(raw)
        if compact:
            data = _expand_compact(data)
    except Exception as exc:
        where = "локальная модель" if cfg.llm.is_local() else "модель"
        return "", [], {}, [f"Смысловая оценка не выполнена ({where}): {exc}"]

    by_id = {r.id: r for r in results}
    for item in data.get("checks", []) or []:
        r = by_id.get(item.get("id"))
        if r is None or r.type != "llm":
            continue
        r.passed = item.get("passed") if isinstance(item.get("passed"), bool) else None
        r.confidence = float(item.get("confidence") or 0.0)
        r.detail = (item.get("reason") or "").strip()[:400]
        q = (item.get("quote") or "").strip()
        if q:
            r.evidence = [_locate(transcript, q)]

    prod_by_name = {p.name: p for p in products}
    for item in data.get("products", []) or []:
        p = prod_by_name.get(item.get("name"))
        if p is None:
            continue
        if item.get("offered") and not p.offered:
            p.offered = True
            p.how = "llm"
            q = (item.get("quote") or "").strip()
            if q:
                p.evidence = [_locate(transcript, q)]

    extra = {
        "llm_quality_score": data.get("quality_score"),
        "client_sentiment": data.get("client_sentiment"),
        "call_outcome": data.get("call_outcome"),
    }
    recs = [str(x) for x in (data.get("recommendations") or [])][:5]
    return str(data.get("summary") or ""), recs, extra, warnings


def _expand_compact(data: Dict[str, Any]) -> Dict[str, Any]:
    """Приводит плоский ответ компактного запроса к общему формату."""
    checks, prods = [], []
    for item in (data.get("answers") or data.get("checks") or []):
        if not isinstance(item, dict):
            continue
        ident = str(item.get("id") or "")
        entry = {
            "passed": item.get("passed") if isinstance(item.get("passed"), bool) else None,
            "confidence": 0.7,
            "reason": (item.get("reason") or "").strip(),
            "quote": (item.get("quote") or "").strip(),
        }
        if ident.startswith("товар::"):
            prods.append({"name": ident.split("::", 1)[1],
                          "offered": bool(entry["passed"]),
                          "confidence": 0.7, "quote": entry["quote"]})
        else:
            checks.append({"id": ident, **entry})
    return {"checks": checks, "products": prods, "summary": "",
            "recommendations": []}


def _locate(transcript: Transcript, quote: str) -> Evidence:
    """Пытается найти таймкод цитаты в стенограмме."""
    from .text import SpeechIndex, find_pattern

    idx = SpeechIndex(transcript.segments)
    ev = find_pattern(idx, quote[:120], fuzzy_threshold=80, fuzzy_min_len=8)
    if ev:
        ev.how = "llm"
        return ev
    return Evidence(quote[:200], 0.0, 0.0, "", "", quote[:60], "llm")


# --------------------------------------------------------------------------- #
def normalize_base_url(url: str) -> str:
    """Достраивает /v1, если пользователь указал только хост.

    Дописывать «/v1» вслепую нельзя: у части провайдеров путь другой
    (у Gemini это /v1beta/openai). Поэтому дополняем только тогда,
    когда пути нет вообще: «http://localhost:11434» -> «.../v1».
    """
    from urllib.parse import urlparse

    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if "//" not in url:                      # «localhost:11434»
        url = "http://" + url
    parsed = urlparse(url)
    if parsed.path in ("", "/"):
        url += "/v1"
    return url


# Провайдеры с бесплатным тарифом, говорящие на протоколе OpenAI.
#
# ВНИМАНИЕ про приватность: на бесплатных тарифах стенограмма звонка уходит
# провайдеру, и многие из них оставляют за собой право использовать её для
# обучения. Для записей с персональными данными клиентов это может быть
# неприемлемо — тогда единственный корректный вариант локальная модель.
FREE_PROVIDERS = [
    {
        "id": "groq",
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "signup": "console.groq.com",
        "limits": "около 30 запросов/мин, тысячи в сутки",
        "note": "самый быстрый ответ, щедрые лимиты",
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
        "signup": "aistudio.google.com/apikey",
        "limits": "около 1500 запросов в сутки, 10 в минуту",
        "note": "лучшее качество на русском среди бесплатных",
    },
    {
        "id": "openrouter",
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-r1:free",
        "key_env": "OPENROUTER_API_KEY",
        "signup": "openrouter.ai/keys",
        "limits": "около 50 запросов в сутки на бесплатных моделях",
        "note": "много моделей на выбор, но лимит суток маленький",
    },
    {
        "id": "cerebras",
        "name": "Cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "model": "gpt-oss-120b",
        "key_env": "CEREBRAS_API_KEY",
        "signup": "cloud.cerebras.ai",
        "limits": "около 5 запросов/мин, 1 млн токенов в сутки",
        "note": "быстро, но жёсткий лимит в минуту",
    },
]


# Локальные серверы, которые понимают протокол OpenAI
LOCAL_SERVERS = [
    ("Ollama", "http://localhost:11434/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
    ("llama.cpp / прочее", "http://localhost:8080/v1"),
    ("Jan / LocalAI", "http://localhost:1337/v1"),
]


def discover_local_servers(timeout: float = 1.5) -> List[Dict[str, Any]]:
    """Ищет запущенные локальные LLM-серверы и их модели."""
    import requests

    found: List[Dict[str, Any]] = []
    for name, url in LOCAL_SERVERS:
        try:
            r = requests.get(url + "/models", timeout=timeout)
            if r.status_code >= 400:
                continue
            models = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
            found.append({"name": name, "base_url": url, "models": models})
        except Exception:
            continue
    return found


def _call(cfg: Config, prompt: str, key: str) -> str:
    import requests

    if cfg.llm.provider == "anthropic":
        url = (cfg.llm.base_url.rstrip("/") + "/v1/messages") if cfg.llm.base_url \
            else "https://api.anthropic.com/v1/messages"
        r = requests.post(
            url,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": cfg.llm.model, "max_tokens": cfg.llm.max_tokens,
                  "temperature": cfg.llm.temperature, "system": SYSTEM,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=cfg.llm.timeout_sec,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Anthropic API {r.status_code}: {r.text[:300]}")
        blocks = r.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    # openai-совместимый (в том числе Ollama, LM Studio, llama.cpp)
    base = normalize_base_url(cfg.llm.base_url) or "https://api.openai.com/v1"
    r = requests.post(
        base + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": cfg.llm.model, "max_tokens": cfg.llm.max_tokens,
              "temperature": cfg.llm.temperature,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": prompt}]},
        timeout=cfg.llm.timeout_sec,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"LLM API {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("модель вернула не JSON")
        return json.loads(m.group(0))


def check_connection(cfg: Config) -> Tuple[bool, str]:
    """Проверка ключа/доступности модели для кнопки в интерфейсе."""
    if not cfg.llm.enabled or cfg.llm.provider == "none":
        return False, "Смысловая оценка выключена в настройках"
    key = cfg.llm_api_key()
    if not key:
        return False, f"Нет ключа: задайте переменную окружения {cfg.llm.api_key_env}"
    try:
        out = _call(cfg, 'Ответь строго: {"ok": true}', key)
        where = " локально" if cfg.llm.is_local() else ""
        return ("ok" in out.lower()), f"Ответ получен{where}: {cfg.llm.model}"
    except Exception as exc:
        msg = str(exc)[:300]
        if cfg.llm.is_local() and ("Connection" in msg or "refused" in msg.lower()):
            return False, ("Локальный сервер не отвечает по адресу "
                           f"{cfg.llm.base_url}. Он запущен?")
        return False, msg
