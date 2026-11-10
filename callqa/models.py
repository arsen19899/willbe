"""Общие структуры данных пайплайна."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

OPERATOR = "operator"
CLIENT = "client"
ROLE_RU = {OPERATOR: "Оператор", CLIENT: "Клиент", "any": "Любой", "unknown": "Неизвестно"}


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    speaker: str          # operator | client | unknown
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def word_count(self) -> int:
        return len(self.words) if self.words else len(self.text.split())


@dataclass
class Transcript:
    segments: List[Segment] = field(default_factory=list)
    duration: float = 0.0
    language: str = "ru"
    channels: int = 1
    source: str = ""          # stereo | diarization | single
    asr_model: str = ""

    def by_speaker(self, speaker: str) -> List[Segment]:
        if speaker == "any":
            return sorted(self.segments, key=lambda s: s.start)
        return [s for s in self.segments if s.speaker == speaker]

    def text_of(self, speaker: str = "any") -> str:
        return " ".join(s.text.strip() for s in self.by_speaker(speaker) if s.text.strip())

    def dialogue(self, max_chars: int = 0) -> str:
        """Читаемая стенограмма с таймкодами и ролями."""
        lines = []
        for s in sorted(self.segments, key=lambda x: x.start):
            if not s.text.strip():
                continue
            lines.append(f"[{_ts(s.start)}] {ROLE_RU.get(s.speaker, s.speaker)}: {s.text.strip()}")
        out = "\n".join(lines)
        if max_chars and len(out) > max_chars:
            head = out[: int(max_chars * 0.6)]
            tail = out[-int(max_chars * 0.35) :]
            out = head + "\n[... середина разговора пропущена ...]\n" + tail
        return out


def _ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


@dataclass
class Evidence:
    """Подтверждение срабатывания: что нашли, где и в чьей речи."""
    quote: str = ""
    start: float = 0.0
    end: float = 0.0
    speaker: str = ""
    pattern: str = ""         # что искали
    matched: str = ""         # что нашли (важно для регулярок и нечёткого поиска)
    how: str = ""             # exact | lemma | fuzzy | regex | metric | llm

    def ts(self) -> str:
        return _ts(self.start)


@dataclass
class CheckResult:
    id: str
    title: str
    category: str
    type: str
    speaker: str
    weight: float
    required: bool
    passed: Optional[bool]          # None = не удалось проверить (н/д)
    detail: str = ""
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 1.0
    invert: bool = False            # пункт пройден, если фразы НЕ найдены

    @property
    def status_ru(self) -> str:
        if self.passed is None:
            return "н/д"
        return "было" if self.passed else "не было"


@dataclass
class ProductResult:
    name: str
    offered: bool
    required: bool
    group: str = ""
    how: str = ""
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class CallResult:
    file: str
    file_name: str = ""
    ok: bool = True
    error: str = ""
    skipped_reason: str = ""
    duration: float = 0.0
    transcript: Optional[Transcript] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    checks: List[CheckResult] = field(default_factory=list)
    products: List[ProductResult] = field(default_factory=list)
    keyword_hits: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    score_max: float = 0.0
    score_pct: float = 0.0
    grade: str = ""
    category_scores: Dict[str, float] = field(default_factory=dict)
    llm_summary: str = ""
    llm_recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_sec: float = 0.0

    def to_dict(self, with_transcript: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not with_transcript:
            d.pop("transcript", None)
        return d
