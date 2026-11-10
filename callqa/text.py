"""Нормализация русского текста и поиск ключевых слов/фраз по транскрипту.

Три уровня поиска, от строгого к мягкому:
  1. точное совпадение нормализованных слов;
  2. совпадение лемм (пришёл/пришла/приходить -> «приходить»);
  3. нечёткое совпадение (опечатки распознавания) через rapidfuzz.

Плюс режим регулярных выражений: шаблон с префиксом ``re:``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from .models import Evidence, Segment

# --------------------------------------------------------------------------- #
#  Морфология (необязательная зависимость)
# --------------------------------------------------------------------------- #
_MORPH = None
_MORPH_TRIED = False


def _morph():
    global _MORPH, _MORPH_TRIED
    if _MORPH_TRIED:
        return _MORPH
    _MORPH_TRIED = True
    for mod in ("pymorphy3", "pymorphy2"):
        try:
            m = __import__(mod)
            _MORPH = m.MorphAnalyzer()
            break
        except Exception:
            continue
    return _MORPH


try:
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover
    _fuzz = None


TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(s: str, yo_to_e: bool = True) -> str:
    s = (s or "").lower().replace(" ", " ")
    if yo_to_e:
        s = s.replace("ё", "е")
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str, yo_to_e: bool = True) -> List[str]:
    return TOKEN_RE.findall(normalize(s, yo_to_e))


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    m = _morph()
    if m is None:
        return word
    try:
        return m.parse(word)[0].normal_form.replace("ё", "е")
    except Exception:
        return word


def lemmatize(tokens: Sequence[str]) -> List[str]:
    return [lemma(t) for t in tokens]


def morphology_available() -> bool:
    return _morph() is not None


def fuzzy_available() -> bool:
    return _fuzz is not None


# --------------------------------------------------------------------------- #
#  Индекс речи одного спикера
# --------------------------------------------------------------------------- #
@dataclass
class Tok:
    norm: str
    lem: str
    start: float
    end: float
    speaker: str
    raw: str
    seg: int = 0      # номер реплики: фраза не должна склеивать соседние реплики


class SpeechIndex:
    """Плоский поток слов с таймкодами + нормализованный текст для regex/fuzzy."""

    def __init__(self, segments: Sequence[Segment], lemmatize_on: bool = True,
                 yo_to_e: bool = True):
        self.toks: List[Tok] = []
        self.lemmatize_on = lemmatize_on
        self.yo_to_e = yo_to_e

        for seg_no, seg in enumerate(sorted(segments, key=lambda s: s.start)):
            if seg.words:
                for w in seg.words:
                    for t in tokenize(w.text, yo_to_e):
                        self.toks.append(
                            Tok(t, lemma(t) if lemmatize_on else t,
                                w.start, w.end, seg.speaker, w.text, seg_no)
                        )
            else:
                # нет пословных таймкодов — распределяем слова равномерно по сегменту
                parts = tokenize(seg.text, yo_to_e)
                n = max(1, len(parts))
                step = seg.duration / n if seg.duration > 0 else 0.0
                for i, t in enumerate(parts):
                    st = seg.start + i * step
                    self.toks.append(
                        Tok(t, lemma(t) if lemmatize_on else t,
                            st, st + step, seg.speaker, t, seg_no)
                    )

        self.norm_tokens = [t.norm for t in self.toks]
        self.lem_tokens = [t.lem for t in self.toks]
        # строка + карта «символ -> индекс токена» для regex и fuzzy
        self.text = " ".join(self.norm_tokens)
        self._char_to_tok: List[int] = []
        for i, t in enumerate(self.norm_tokens):
            self._char_to_tok.extend([i] * len(t))
            if i < len(self.norm_tokens) - 1:
                self._char_to_tok.append(i)  # пробел относим к предыдущему токену

    # ---------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.toks)

    def span_time(self, i: int, j: int) -> Tuple[float, float]:
        if not self.toks:
            return 0.0, 0.0
        i = max(0, min(i, len(self.toks) - 1))
        j = max(0, min(j, len(self.toks) - 1))
        return self.toks[i].start, self.toks[j].end

    def quote(self, i: int, j: int, pad: int = 6, max_chars: int = 180) -> str:
        a = max(0, i - pad)
        b = min(len(self.toks), j + 1 + pad)
        words = [t.raw for t in self.toks[a:b]]
        q = " ".join(words).strip()
        if a > 0:
            q = "… " + q
        if b < len(self.toks):
            q = q + " …"
        return q[:max_chars]

    def matched_text(self, i: int, j: int, max_chars: int = 60) -> str:
        """Именно то, что совпало — без окружающих слов."""
        a = max(0, min(i, len(self.toks) - 1))
        b = max(0, min(j, len(self.toks) - 1))
        text = " ".join(t.raw for t in self.toks[a:b + 1]).strip()
        return text[:max_chars]

    def tok_at_char(self, pos: int) -> int:
        if not self._char_to_tok:
            return 0
        pos = max(0, min(pos, len(self._char_to_tok) - 1))
        return self._char_to_tok[pos]

    def slice_by_time(self, t_from: Optional[float], t_to: Optional[float]) -> "SpeechIndex":
        """Подиндекс по временному окну (для проверок вида «первые 40 секунд»)."""
        if t_from is None and t_to is None:
            return self
        lo = t_from if t_from is not None else -1e9
        hi = t_to if t_to is not None else 1e9
        sub = SpeechIndex.__new__(SpeechIndex)
        sub.lemmatize_on = self.lemmatize_on
        sub.yo_to_e = self.yo_to_e
        sub.toks = [t for t in self.toks if t.start >= lo and t.start <= hi]
        sub.norm_tokens = [t.norm for t in sub.toks]
        sub.lem_tokens = [t.lem for t in sub.toks]
        sub.text = " ".join(sub.norm_tokens)
        sub._char_to_tok = []
        for i, t in enumerate(sub.norm_tokens):
            sub._char_to_tok.extend([i] * len(t))
            if i < len(sub.norm_tokens) - 1:
                sub._char_to_tok.append(i)
        return sub


# --------------------------------------------------------------------------- #
#  Поиск
# --------------------------------------------------------------------------- #
def _seq_search(haystack: List[str], needle: List[str], max_gap: int = 0,
                start: int = 0) -> Optional[Tuple[int, int]]:
    """Ищет последовательность слов, допуская до max_gap вставок между словами."""
    if not needle or not haystack:
        return None
    n = len(haystack)
    first = needle[0]
    for i in range(max(0, start), n):
        if haystack[i] != first:
            continue
        pos = i
        ok = True
        for w in needle[1:]:
            found = -1
            for j in range(pos + 1, min(n, pos + 2 + max_gap)):
                if haystack[j] == w:
                    found = j
                    break
            if found < 0:
                ok = False
                break
            pos = found
        if ok:
            return i, pos
    return None


def find_pattern(index: SpeechIndex, pattern: str, fuzzy_threshold: int = 88,
                 fuzzy_min_len: int = 8, max_gap_words: int = 2,
                 evidence_chars: int = 180) -> Optional[Evidence]:
    """Возвращает первое совпадение шаблона в речи или None."""
    if not pattern or not len(index):
        return None
    pattern = pattern.strip()

    # --- регулярное выражение ---------------------------------------- #
    if pattern.lower().startswith("re:"):
        rx = pattern[3:].strip()
        try:
            m = re.search(rx, index.text, re.IGNORECASE | re.UNICODE)
        except re.error as exc:
            raise ValueError(f"Неверное регулярное выражение '{rx}': {exc}") from exc
        if not m:
            return None
        i = index.tok_at_char(m.start())
        j = index.tok_at_char(max(m.start(), m.end() - 1))
        st, en = index.span_time(i, j)
        return Evidence(index.quote(i, j, max_chars=evidence_chars), st, en,
                        index.toks[i].speaker if index.toks else "", pattern,
                        index.matched_text(i, j), "regex")

    words = tokenize(pattern, index.yo_to_e)
    if not words:
        return None

    # --- точное совпадение ------------------------------------------- #
    hit = _seq_search(index.norm_tokens, words, 0)
    how = "exact"
    # --- по леммам ---------------------------------------------------- #
    if hit is None and index.lemmatize_on:
        hit = _seq_search(index.lem_tokens, lemmatize(words), max_gap_words)
        how = "lemma"
    # --- нечёткое ----------------------------------------------------- #
    if hit is None and fuzzy_threshold and _fuzz is not None:
        pat_norm = " ".join(words)
        if len(pat_norm) >= fuzzy_min_len:
            hit_span = _fuzzy_span(index.text, pat_norm, fuzzy_threshold)
            if hit_span:
                i = index.tok_at_char(hit_span[0])
                j = index.tok_at_char(max(hit_span[0], hit_span[1] - 1))
                hit, how = (i, j), "fuzzy"

    if hit is None:
        return None
    i, j = hit
    st, en = index.span_time(i, j)
    return Evidence(index.quote(i, j, max_chars=evidence_chars), st, en,
                    index.toks[i].speaker if index.toks else "", pattern,
                    index.matched_text(i, j), how)


def _fuzzy_span(text: str, pattern: str, threshold: int) -> Optional[Tuple[int, int]]:
    if _fuzz is None or not text:
        return None
    try:
        al = _fuzz.partial_ratio_alignment(pattern, text, score_cutoff=threshold)
    except Exception:
        score = _fuzz.partial_ratio(pattern, text)
        return (0, len(text)) if score >= threshold else None
    if al is None:
        return None
    return al.dest_start, al.dest_end


def find_all(index: SpeechIndex, patterns: Sequence[str], **kw) -> List[Evidence]:
    out: List[Evidence] = []
    for p in patterns:
        ev = find_pattern(index, p, **kw)
        if ev:
            out.append(ev)
    return out


def find_occurrences(index: SpeechIndex, pattern: str, max_hits: int = 30,
                     max_gap_words: int = 2, evidence_chars: int = 180
                     ) -> List[Evidence]:
    """Все вхождения шаблона, а не только первое.

    Нужно для проверок вида «не меньше трёх вежливых оборотов» и для поиска
    каждого возражения клиента, чтобы проверить реакцию оператора на каждое.
    Нечёткий поиск здесь не применяется — он даёт одно совпадение и на счётчиках
    вёл бы себя непредсказуемо.
    """
    out: List[Evidence] = []
    if not pattern or not len(index):
        return out
    pattern = pattern.strip()

    if pattern.lower().startswith("re:"):
        rx = pattern[3:].strip()
        try:
            for m in re.finditer(rx, index.text, re.IGNORECASE | re.UNICODE):
                i = index.tok_at_char(m.start())
                j = index.tok_at_char(max(m.start(), m.end() - 1))
                st, en = index.span_time(i, j)
                out.append(Evidence(index.quote(i, j, max_chars=evidence_chars),
                                    st, en, index.toks[i].speaker, pattern,
                                    index.matched_text(i, j), "regex"))
                if len(out) >= max_hits:
                    break
        except re.error:
            return out
        return out

    words = tokenize(pattern, index.yo_to_e)
    if not words:
        return out

    for tokens, how, gap in ((index.norm_tokens, "exact", 0),
                             (index.lem_tokens, "lemma", max_gap_words)):
        if how == "lemma" and not index.lemmatize_on:
            break
        needle = words if how == "exact" else lemmatize(words)
        pos = 0
        found_here: List[Evidence] = []
        while len(found_here) < max_hits:
            hit = _seq_search(tokens, needle, gap, start=pos)
            if hit is None:
                break
            i, j = hit
            st, en = index.span_time(i, j)
            found_here.append(Evidence(index.quote(i, j, max_chars=evidence_chars),
                                       st, en, index.toks[i].speaker, pattern,
                                       index.matched_text(i, j), how))
            pos = j + 1
        if found_here:
            return found_here
    return out


def count_occurrences(index: SpeechIndex, patterns: Sequence[str],
                      max_gap_words: int = 2) -> Tuple[int, List[Evidence]]:
    """Сколько всего раз встретились фразы из списка."""
    hits: List[Evidence] = []
    for p in patterns:
        hits.extend(find_occurrences(index, p, max_gap_words=max_gap_words))
    hits.sort(key=lambda e: e.start)
    return len(hits), hits
