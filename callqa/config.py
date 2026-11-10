"""Конфигурация CallQA: загрузка/сохранение/валидация YAML-настроек."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default.yaml"
)


# --------------------------------------------------------------------------- #
#  Вспомогательное
# --------------------------------------------------------------------------- #
def _merge(base: dict, override: dict) -> dict:
    """Рекурсивно накладывает override на base (списки заменяются целиком)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# --------------------------------------------------------------------------- #
#  Секции
# --------------------------------------------------------------------------- #
@dataclass
class ASRConfig:
    engine: str = "faster-whisper"
    model: str = "large-v3"          # tiny|base|small|medium|large-v3 или путь
    device: str = "auto"             # auto|cuda|cpu
    compute_type: str = "auto"       # auto|float16|int8_float16|int8
    language: str = "ru"
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True
    initial_prompt: str = ""
    cpu_threads: int = 0             # 0 = авто
    download_root: str = ""          # куда качать модели (пусто = кэш HF)


@dataclass
class SpeakersConfig:
    mode: str = "auto"               # auto|stereo|diarize|single
    # какая дорожка чья, если файл стерео
    left: str = "operator"
    right: str = "client"
    # автоопределение ролей по репликам (перебивает left/right, если уверенность высокая)
    autodetect_roles: bool = True
    operator_markers: List[str] = field(default_factory=list)
    client_markers: List[str] = field(default_factory=list)
    # если корреляция каналов выше — считаем файл фактически моно
    dual_mono_corr: float = 0.98
    # диаризация (для моно)
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    hf_token: str = ""
    num_speakers: int = 2


@dataclass
class MatchingConfig:
    lemmatize: bool = True           # приводить слова к начальной форме (pymorphy3)
    yo_to_e: bool = True             # ё -> е
    fuzzy_threshold: int = 88        # 0 = отключить нечёткий поиск, иначе 70..100
    fuzzy_min_len: int = 8           # не «фаззить» слишком короткие фразы
    max_gap_words: int = 2           # сколько слов можно вставить внутрь фразы


@dataclass
class MetricsConfig:
    interruption_overlap_sec: float = 0.6   # минимальное наложение реплик
    interruption_min_words: int = 2         # реплика короче — не перебивание, а поддакивание
    interruption_ignore_backchannel: bool = True
    backchannel_words: List[str] = field(default_factory=list)
    long_pause_sec: float = 3.0             # «тишина в эфире»
    monologue_sec: float = 45.0             # непрерывный монолог одной стороны
    fast_speech_wpm: float = 180.0
    slow_speech_wpm: float = 90.0


@dataclass
class LLMConfig:
    enabled: bool = True
    provider: str = "anthropic"      # anthropic|openai|none
    model: str = "claude-sonnet-5"
    api_key: str = ""                # лучше оставить пустым и задать переменную окружения
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = ""               # для openai-совместимых шлюзов
    max_tokens: int = 4000
    temperature: float = 0.0
    timeout_sec: int = 120
    # компактный запрос: только вердикты по пунктам, без резюме и рекомендаций.
    # auto = включать для локальных моделей (они слабее и медленнее)
    compact: str = "auto"           # auto|true|false

    def is_local(self) -> bool:
        """Локальный сервер (Ollama, LM Studio, llama.cpp) — ключ не нужен."""
        url = (self.base_url or "").lower()
        return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1",
                                      "host.docker.internal"))

    def use_compact(self) -> bool:
        if self.compact == "true":
            return True
        if self.compact == "false":
            return False
        return self.is_local()


@dataclass
class FiltersConfig:
    """Отбор файлов и «красные флаги» в разговоре."""
    min_duration_sec: float = 10.0
    max_duration_sec: float = 7200.0
    min_words: int = 10
    skip_if_silence_ratio_above: float = 0.97
    file_name_include: List[str] = field(default_factory=list)   # подстроки/regex re:
    file_name_exclude: List[str] = field(default_factory=list)
    stop_phrases: List[str] = field(default_factory=list)        # запрещённые фразы оператора
    profanity: List[str] = field(default_factory=list)


@dataclass
class OutputConfig:
    dir: str = "reports"
    formats: List[str] = field(default_factory=lambda: ["xlsx", "csv", "json"])
    save_transcript: bool = True
    transcript_in_report: bool = True
    evidence_chars: int = 180


@dataclass
class Product:
    name: str
    aliases: List[str] = field(default_factory=list)
    required: bool = False           # обязателен ли к предложению
    group: str = ""


@dataclass
class Check:
    """Один пункт чек-листа."""
    id: str
    title: str
    type: str = "keyword"            # keyword|regex|sequence|metric|llm|product
    category: str = "Общее"
    speaker: str = "operator"        # operator|client|any
    weight: float = 1.0
    required: bool = False           # критичный пункт: провал -> отметка в своде
    invert: bool = False             # True: пункт пройден, если НЕ найдено
    any_of: List[str] = field(default_factory=list)
    all_of: List[str] = field(default_factory=list)
    none_of: List[str] = field(default_factory=list)
    min_hits: int = 1                # сколько раз должно встретиться (для any_of)
    time_from: Optional[float] = None
    time_to: Optional[float] = None
    # для type=sequence — «реплика одного, реакция другого»
    trigger_speaker: str = "client"
    trigger_any_of: List[str] = field(default_factory=list)
    within_sec: float = 30.0
    if_no_trigger: str = "pass"      # pass|fail|na — если повода не было
    # для type=metric
    metric: str = ""
    op: str = "<="                   # <= < >= > == !=
    value: float = 0.0
    # для type=llm
    prompt: str = ""
    comment: str = ""


@dataclass
class Config:
    version: int = 1
    asr: ASRConfig = field(default_factory=ASRConfig)
    speakers: SpeakersConfig = field(default_factory=SpeakersConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    products: List[Product] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)
    # заполняется validate(); в YAML не сохраняется
    warnings: List[str] = field(default_factory=list, compare=False, repr=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Config":
        data = data or {}
        cfg = Config(
            version=int(data.get("version", 1)),
            asr=_build(ASRConfig, data.get("asr")),
            speakers=_build(SpeakersConfig, data.get("speakers")),
            matching=_build(MatchingConfig, data.get("matching")),
            metrics=_build(MetricsConfig, data.get("metrics")),
            llm=_build(LLMConfig, data.get("llm")),
            filters=_build(FiltersConfig, data.get("filters")),
            output=_build(OutputConfig, data.get("output")),
            products=[_build(Product, p) for p in (data.get("products") or [])],
            checks=[_build(Check, c) for c in (data.get("checks") or [])],
        )
        cfg.warnings = cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("warnings", None)      # служебное поле, в файл не пишем
        return d

    @staticmethod
    def load(path: Optional[str] = None) -> "Config":
        """Читает YAML. Если передан пользовательский файл — накладывает его на дефолтный."""
        base: Dict[str, Any] = {}
        if os.path.exists(DEFAULT_CONFIG_PATH):
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}
        if path and os.path.abspath(path) != os.path.abspath(DEFAULT_CONFIG_PATH):
            with open(path, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            base = _merge(base, user)
        return Config.from_dict(base)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(), f, allow_unicode=True, sort_keys=False, width=100
            )

    # ------------------------------------------------------------------ #
    def validate(self) -> List[str]:
        """Проверяет настройки.

        Ошибки, которые сделают разбор бессмысленным, поднимают исключение.
        Незаполненные пункты — не ошибка: их можно дозаполнить позже,
        в отчёте они будут помечены «н/д». Такие случаи возвращаются
        списком предупреждений, чтобы интерфейс их показал.
        """
        warnings: List[str] = []
        seen: Dict[str, str] = {}
        for c in self.checks:
            label = c.title or c.id or "(без названия)"
            if not c.id:
                raise ValueError(f"«{label}»: отсутствует id")
            if c.id in seen:
                raise ValueError(
                    f"Два пункта имеют одинаковый id «{c.id}»: "
                    f"«{seen[c.id]}» и «{label}»"
                )
            seen[c.id] = label
            if c.type not in ("keyword", "regex", "sequence", "metric", "llm", "product"):
                raise ValueError(f"«{label}»: неизвестный тип проверки «{c.type}»")
            if c.speaker not in ("operator", "client", "any"):
                raise ValueError(f"«{label}»: чья речь — operator, client или any")
            if c.type in ("keyword", "regex") and not (c.any_of or c.all_of or c.none_of):
                warnings.append(f"«{label}»: не заданы фразы — пункт будет «н/д»")
            if c.type == "sequence":
                if not c.trigger_any_of:
                    warnings.append(f"«{label}»: не заданы реплики-поводы — пункт будет «н/д»")
                if not c.any_of:
                    warnings.append(f"«{label}»: не заданы фразы ответа — пункт будет «н/д»")
                if c.if_no_trigger not in ("pass", "fail", "na"):
                    raise ValueError(f"«{label}»: if_no_trigger — pass, fail или na")
            if c.type == "metric" and not c.metric:
                warnings.append(f"«{label}»: не выбрана метрика — пункт будет «н/д»")
            if c.type == "llm" and not (c.prompt or c.title):
                warnings.append(f"«{label}»: не задан вопрос к модели — пункт будет «н/д»")

        names: Dict[str, int] = {}
        for p in self.products:
            if not p.name:
                warnings.append("У товара пустое название — он пропущен")
                continue
            names[p.name] = names.get(p.name, 0) + 1
        for name, count in names.items():
            if count > 1:
                raise ValueError(f"Товар «{name}» указан {count} раза — оставьте один")

        if self.speakers.mode not in ("auto", "stereo", "diarize", "single"):
            raise ValueError("speakers.mode: auto, stereo, diarize или single")
        return warnings

    def llm_api_key(self) -> str:
        key = self.llm.api_key or os.environ.get(self.llm.api_key_env, "")
        if not key and self.llm.is_local():
            return "local"      # локальные серверы ключ требуют, но не проверяют
        return key


def _build(cls, data):
    """Создаёт dataclass, игнорируя неизвестные ключи (чтобы конфиг не ломался при апдейтах)."""
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"Ожидался словарь для {cls.__name__}, получено {type(data)}")
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in data.items() if k in fields}
    unknown = set(data) - fields
    if unknown:
        # мягко: не падаем, но и не молчим
        import warnings

        warnings.warn(f"{cls.__name__}: неизвестные поля игнорируются: {sorted(unknown)}")
    return cls(**kwargs)
