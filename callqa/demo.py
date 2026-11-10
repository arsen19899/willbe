"""Демо-разговор: даёт увидеть формат отчёта до установки моделей распознавания.

Стенограмма собрана вручную и намеренно содержит и хорошее (приветствие,
выявление потребности, отработку возражения), и плохое (перебивание клиента,
не спросили имя, не предложили страховку).
"""
from __future__ import annotations

from typing import List

from .config import Config
from .models import CLIENT, OPERATOR, CallResult, Segment, Transcript, Word

_DIALOGUE = [
    (OPERATOR, 0.0, "Здравствуйте меня зовут Игорь компания Светотехника чем могу помочь"),
    (CLIENT, 7.0, "Добрый день я хотел бы купить настольную лампу для работы"),
    (OPERATOR, 12.5, "Подскажите пожалуйста для каких целей вам нужна лампа и какой у вас бюджет"),
    (CLIENT, 18.0, "Ну я работаю за компьютером вечерами хотелось бы что-то не очень дорогое"),
    (OPERATOR, 22.0, "Понял вас у нас есть отличная модель она стоит 4500 рублей"),
    (CLIENT, 28.5, "А что там по гарантии"),
    (OPERATOR, 31.0, "Стандартная гарантия год но я рекомендую оформить "
                     "расширенную гарантию на два года"),
    (CLIENT, 38.0, "Дороговато выходит я подумаю"),
    (OPERATOR, 41.5, "Понимаю ваши сомнения зато вы будете спокойны два года "
                     "и любой ремонт будет бесплатным"),
    (CLIENT, 48.5, "Хорошо а доставка есть"),
    (OPERATOR, 51.0, "Конечно доставка курьером завтра до двери оформляем заказ"),
    (CLIENT, 56.5, "Давайте оформляйте"),
    (OPERATOR, 59.0, "Отлично записываю вас спасибо за обращение хорошего дня до свидания"),
]


def _seg(spk: str, start: float, text: str, wps: float = 2.6) -> Segment:
    step = 1.0 / wps
    words, t = [], start
    for w in text.split():
        words.append(Word(w, round(t, 2), round(t + step, 2)))
        t += step
    return Segment(spk, start, round(t, 2), text, words)


def demo_transcript() -> Transcript:
    return Transcript(
        segments=[_seg(s, t, x) for s, t, x in _DIALOGUE],
        duration=65.0, channels=2, source="stereo", asr_model="demo",
    )


def run(cfg: Config = None, use_llm: bool = False) -> List[CallResult]:
    from .pipeline import analyze_file

    cfg = cfg or Config.load()
    if not use_llm:
        cfg.llm.enabled = False
    r = analyze_file("demo_zvonok.wav", cfg, transcript_override=demo_transcript())
    r.file_name = "demo_zvonok.wav (пример)"
    r.warnings.insert(0, "Это демонстрационный разбор на встроенной стенограмме — "
                         "аудио не распознавалось.")
    return [r]
