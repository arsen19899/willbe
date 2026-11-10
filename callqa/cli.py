"""Командная строка CallQA."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from . import audio as au, report
from .config import DEFAULT_CONFIG_PATH, Config
from .models import CallResult
from .pipeline import analyze_many


def _progress(msg: str, frac: float) -> None:
    bar = int(frac * 30)
    sys.stderr.write(f"\r[{'█' * bar}{'·' * (30 - bar)}] {msg[:70]:<70}")
    sys.stderr.flush()


def cmd_analyze(args) -> int:
    cfg = Config.load(args.config)
    if args.model:
        cfg.asr.model = args.model
    if args.device:
        cfg.asr.device = args.device
    if args.no_llm:
        cfg.llm.enabled = False
    if args.out:
        cfg.output.dir = args.out

    paths = []
    for src in args.input:
        if not os.path.exists(src):
            print(f"Не найдено: {src}", file=sys.stderr)
            return 2
        paths.extend(au.list_audio_files(src, recursive=not args.no_recursive))
    paths = sorted(set(paths))
    if not paths:
        print("Аудиофайлы не найдены.", file=sys.stderr)
        return 2

    print(f"Файлов к разбору: {len(paths)}", file=sys.stderr)
    results = analyze_many(paths, cfg, progress=None if args.quiet else _progress)
    sys.stderr.write("\n")

    stem = args.name or f"callqa_{datetime.now():%Y%m%d_%H%M%S}"
    written = report.write_all(results, cfg, cfg.output.dir, stem)

    _print_summary(results)
    for fmt, p in written.items():
        print(f"{fmt}: {p}")
    return 0 if all(r.ok for r in results) else 1


def _print_summary(results) -> None:
    ok = [r for r in results if r.ok]
    print()
    print(f"{'Файл':<38} {'Балл':>6}  {'Оценка':<18} {'Перебив.':>8}  Провалы")
    print("-" * 90)
    for r in results:
        if not r.ok:
            reason = r.skipped_reason or r.error
            print(f"{r.file_name[:37]:<38} {'—':>6}  {reason[:40]}")
            continue
        fails = [c.title for c in r.checks if c.required and c.passed is False]
        print(f"{r.file_name[:37]:<38} {r.score_pct:>5.1f}%  {r.grade:<18} "
              f"{r.metrics.get('operator_interruptions', 0):>8}  "
              f"{', '.join(fails)[:40]}")
    if ok:
        print("-" * 90)
        print(f"Средний балл: {sum(r.score_pct for r in ok) / len(ok):.1f}%")


def cmd_demo(args) -> int:
    from . import demo

    cfg = Config.load(args.config)
    if args.out:
        cfg.output.dir = args.out
    results = demo.run(cfg, use_llm=args.llm)
    written = report.write_all(results, cfg, cfg.output.dir, "callqa_demo")
    _print_summary(results)
    print("\nПодробности по пунктам:")
    for c in results[0].checks:
        ev = c.evidence[0].quote if c.evidence else ""
        print(f"  {c.status_ru:<8} {c.title:<46} {c.detail[:50]}")
        if ev:
            print(f"           └ «{ev[:70]}»")
    print("\nТовары:")
    for p in results[0].products:
        print(f"  {'предложен' if p.offered else 'НЕ предложен':<14} {p.name}")
    print()
    for fmt, p in written.items():
        print(f"{fmt}: {p}")
    return 0


def cmd_learn(args) -> int:
    """Обучение правил на размеченных записях."""
    import yaml

    from . import learn

    cfg = Config.load(args.config)
    if args.model:
        cfg.asr.model = args.model
    cfg.llm.enabled = False        # для обучения смысловая оценка не нужна

    for folder in (args.good, args.bad):
        if not os.path.isdir(folder):
            print(f"Не папка: {folder}", file=sys.stderr)
            return 2

    res = learn.analyse(args.good, args.bad, cfg,
                        progress=None if args.quiet else _progress,
                        min_calls=args.min_calls, max_q=args.max_q,
                        min_lift=args.min_lift)
    sys.stderr.write("\n")

    print(f"Разобрано: успешных {res.good_total}, неуспешных {res.bad_total}")
    if res.failed:
        print(f"Не разобрано файлов: {len(res.failed)}")
        for f in res.failed[:5]:
            print(f"  {f}")

    for w in res.warnings:
        print(f"\n[!] {w}")

    v = res.validation
    if v.get("ok"):
        print(f"\nПроверка на отложенных записях: точность {v['accuracy']:.0%} "
              f"(угадывание большей группы дало бы {v['baseline']:.0%}), "
              f"AUC {v['auc']}")

    if res.phrases:
        print(f"\nОтличающие фразы ({len(res.phrases)}):")
        print(f"{'':2} {'фраза':<34} {'кто':<9} {'успеш.':>7} {'неусп.':>7} "
              f"{'раз':>5} {'q':>7}")
        print("-" * 82)
        for i, p in enumerate(res.phrases):
            mark = "!" if p.outcome_risk else ("+" if p.direction == learn.GOOD else "-")
            who = "оператор" if p.speaker == "operator" else "клиент"
            print(f"{mark:2} {p.display[:33]:<34} {who:<9} "
                  f"{p.good_rate:>6.0%} {p.bad_rate:>7.0%} "
                  f"{p.lift:>5.1f} {p.q_value:>7.3f}")
        print("\n+ признак успеха · - признак неудачи · "
              "! вероятно следствие, а не причина")

    if res.metrics:
        print(f"\nОтличающие метрики ({len(res.metrics)}):")
        for m in res.metrics:
            print(f"  {m.title}: у успешных {m.good_mean:g}, у неуспешных "
                  f"{m.bad_mean:g} -> порог {m.op} {m.threshold:g} "
                  f"(точность {m.accuracy:.0%}, AUC {m.auc:g})")

    checks = learn.build_checks(res)
    if not checks:
        print("\nПредлагать нечего.")
        return 0

    out = args.out or "learned_checks.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump({"checks": checks}, f, allow_unicode=True,
                       sort_keys=False, width=100)
    print(f"\nПредложено пунктов: {len(checks)} -> {out}")

    if args.apply:
        merged = learn.merge_into_config(cfg, checks)
        target = args.apply if isinstance(args.apply, str) else \
            os.path.join(os.path.dirname(DEFAULT_CONFIG_PATH), "current.yaml")
        merged.save(target)
        print(f"Добавлены к чек-листу: {target}")
    else:
        print("Чтобы добавить их в чек-лист: тот же запуск с --apply "
              "(или вкладка «Обучение» в веб-интерфейсе).")
    return 0


def cmd_web(args) -> int:
    from .web.app import run

    run(host=args.host, port=args.port, config_path=args.config, open_browser=not args.no_open)
    return 0


def cmd_config(args) -> int:
    cfg = Config.load(args.config)
    if args.init:
        target = args.init
        cfg.save(target)
        print(f"Конфиг сохранён: {target}")
        return 0
    print(f"Пунктов чек-листа: {len(cfg.checks)}")
    for c in cfg.checks:
        print(f"  [{c.type:<7}] {c.id:<20} вес {c.weight:<5} {c.title}")
    print(f"\nТоваров: {len(cfg.products)}")
    for p in cfg.products:
        mark = "обязателен" if p.required else ""
        print(f"  {p.name:<28} {mark}")
    return 0


def _gpu_hardware() -> str:
    """Имя видеокарты NVIDIA по данным драйвера — без установленного torch."""
    import shutil as _sh
    import subprocess as _sp

    exe = _sh.which("nvidia-smi")
    if not exe and os.name == "nt":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                            "System32", "nvidia-smi.exe")
        if os.path.isfile(cand):
            exe = cand
    if not exe:
        return ""
    try:
        r = _sp.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()[0].strip() if r.returncode == 0 and r.stdout.strip() else ""
    except Exception:
        return ""


def cmd_doctor(args) -> int:
    """Проверка окружения: что установлено, что нет."""
    from . import llm as llm_mod
    from .text import fuzzy_available, morphology_available

    cfg = Config.load(args.config)
    rows = []

    def probe(name, fn):
        try:
            ok, note = fn()
        except Exception as exc:
            ok, note = False, str(exc)[:120]
        rows.append((name, ok, note))

    def _fw():
        import faster_whisper  # noqa
        return True, f"модель в настройках: {cfg.asr.model}"

    def _torch():
        import torch
        if torch.cuda.is_available():
            return True, "CUDA работает: " + torch.cuda.get_device_name(0)
        gpu = _gpu_hardware()
        if gpu:
            return False, (f"видеокарта есть ({gpu}), но torch её не видит — "
                           "запустите install_extras.bat")
        return True, "только CPU (видеокарты NVIDIA нет, распознавание будет медленным)"

    def _pyannote():
        import pyannote.audio  # noqa
        return True, ("токен задан" if cfg.speakers.hf_token
                      else "нет hf_token — для моно будет упрощённая диаризация")

    def _ffmpeg():
        p = au.ffmpeg_path()
        return bool(p), p or "не найден — запустите install_extras.bat"

    probe("faster-whisper", _fw)
    probe("torch / GPU", _torch)
    probe("ffmpeg", _ffmpeg)
    probe("морфология (pymorphy)", lambda: (morphology_available(), "поиск по формам слов"))
    probe("нечёткий поиск (rapidfuzz)", lambda: (fuzzy_available(), "устойчивость к опечаткам"))
    probe("диаризация (pyannote)", _pyannote)
    probe("openpyxl", lambda: (__import__("openpyxl") and True, "выгрузка Excel"))
    if cfg.llm.enabled:
        probe("LLM", lambda: llm_mod.check_connection(cfg))

    print(f"{'Компонент':<28} {'':3} Комментарий")
    print("-" * 92)
    for name, ok, note in rows:
        print(f"{name:<28} {'OK ' if ok else 'нет'} {note}")
    missing = [n for n, ok, _ in rows if not ok]
    if missing:
        print("-" * 92)
        print("Доустановить недостающее:  install_extras.bat")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="callqa",
        description="Анализ записей звонков: ключевые фразы, перебивания, "
                    "предложенные товары, оценка качества диалога.",
    )
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH,
                   help="путь к YAML-конфигу (по умолчанию config/default.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="разобрать файл или папку")
    a.add_argument("input", nargs="+", help="файлы и/или папки с записями")
    a.add_argument("-o", "--out", help="папка для отчётов")
    a.add_argument("-n", "--name", help="имя файла отчёта без расширения")
    a.add_argument("--model", help="переопределить модель whisper")
    a.add_argument("--device", choices=["auto", "cpu", "cuda"])
    a.add_argument("--no-llm", action="store_true", help="без смысловой оценки моделью")
    a.add_argument("--no-recursive", action="store_true", help="не заходить в подпапки")
    a.add_argument("-q", "--quiet", action="store_true")
    a.set_defaults(func=cmd_analyze)

    w = sub.add_parser("web", help="запустить локальный веб-интерфейс")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-open", action="store_true", help="не открывать браузер")
    w.set_defaults(func=cmd_web)

    c = sub.add_parser("config", help="показать или создать конфиг")
    c.add_argument("--init", metavar="ПУТЬ", help="сохранить копию конфига")
    c.set_defaults(func=cmd_config)

    ln = sub.add_parser("learn", help="обучить правила на размеченных записях")
    ln.add_argument("--good", required=True, metavar="ПАПКА",
                    help="папка с успешными разговорами")
    ln.add_argument("--bad", required=True, metavar="ПАПКА",
                    help="папка с неуспешными разговорами")
    ln.add_argument("-o", "--out", help="куда сохранить предложенные пункты")
    ln.add_argument("--apply", nargs="?", const=True, metavar="КОНФИГ",
                    help="сразу добавить пункты в чек-лист")
    ln.add_argument("--min-calls", type=int, default=3,
                    help="минимум записей, где встречается фраза (по умолчанию 3)")
    ln.add_argument("--max-q", type=float, default=0.10,
                    help="порог значимости с поправкой (по умолчанию 0.10)")
    ln.add_argument("--min-lift", type=float, default=1.6,
                    help="во сколько раз чаще должна встречаться фраза")
    ln.add_argument("--model", help="переопределить модель whisper")
    ln.add_argument("-q", "--quiet", action="store_true")
    ln.set_defaults(func=cmd_learn)

    m = sub.add_parser("demo", help="разобрать встроенный пример без аудио и моделей")
    m.add_argument("-o", "--out", help="папка для отчёта")
    m.add_argument("--llm", action="store_true", help="включить смысловую оценку")
    m.set_defaults(func=cmd_demo)

    d = sub.add_parser("doctor", help="проверить окружение")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
