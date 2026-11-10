"""Тесты установщика: определение версии Python, создание окружения, отчёт об ошибках.

Проверяется главное, на чём споткнулась прошлая версия: подходящий интерпретатор
определяется по фактически напечатанной версии, а не по коду возврата (новый
Python Install Manager отдаёт 0 даже когда рантайма нет), а созданное окружение
проверяется по наличию файла python.exe.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import install as inst  # noqa: E402


def fake_python(dirpath: Path, name: str, version: str, exit_code: int = 0,
                broken: bool = False) -> Path:
    """Поддельный интерпретатор: печатает заданную версию на запрос -c.

    broken=True воспроизводит поведение нового py-лаунчера: ругается в stderr,
    ничего не печатает в stdout, но возвращает код 0.
    """
    p = dirpath / name
    body = f'''#!{sys.executable}
import sys
if {broken!r}:
    sys.stderr.write("No runtime installed that matches 3.12\\n")
    sys.exit({exit_code})
if "-c" in sys.argv:
    src = sys.argv[sys.argv.index("-c") + 1]
    src = src.replace("sys.version_info[0]", "{version.split('.')[0]}")
    src = src.replace("sys.version_info[1]", "{version.split('.')[1]}")
    exec(compile(src, "<probe>", "exec"), {{"sys": sys}})
sys.exit({exit_code})
'''
    p.write_text(body, encoding="utf-8")
    p.chmod(0o755)
    return p


class TestProbe(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_reports_real_version(self):
        p = fake_python(self.d, "py312", "3.12")
        self.assertEqual(inst.probe([str(p)]), (3, 12))

    def test_liar_launcher_rejected(self):
        """Главный баг: код возврата 0, но версия не напечатана."""
        p = fake_python(self.d, "pyliar", "3.12", exit_code=0, broken=True)
        self.assertIsNone(inst.probe([str(p)]),
                          "интерпретатор без вывода не должен приниматься")

    def test_nonzero_exit_but_prints_version(self):
        """Печатает версию, но выходит с ненулевым кодом — принимаем."""
        p = fake_python(self.d, "pyodd", "3.11", exit_code=3)
        self.assertEqual(inst.probe([str(p)]), (3, 11))

    def test_missing_binary(self):
        self.assertIsNone(inst.probe([str(self.d / "нет-такого")]))

    def test_store_stub_rejected(self):
        """Заглушка Microsoft Store печатает свой текст, а не версию."""
        p = self.d / "pystub"
        p.write_text(f'#!{sys.executable}\nprint("Python was not found; run without '
                     f'arguments to install from the Microsoft Store")\n', encoding="utf-8")
        p.chmod(0o755)
        self.assertIsNone(inst.probe([str(p)]))


class TestVersionGate(unittest.TestCase):
    def test_supported_range(self):
        self.assertEqual(inst.MIN_VER, (3, 10))
        self.assertEqual(inst.MAX_VER, (3, 12))

    def test_314_is_out_of_range(self):
        self.assertFalse(inst.MIN_VER <= (3, 14) <= inst.MAX_VER)

    def test_313_is_out_of_range(self):
        self.assertFalse(inst.MIN_VER <= (3, 13) <= inst.MAX_VER)

    def test_312_and_310_are_in_range(self):
        for v in ((3, 10), (3, 11), (3, 12)):
            self.assertTrue(inst.MIN_VER <= v <= inst.MAX_VER, v)


class TestFindPython(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self._orig = inst.candidates

    def tearDown(self):
        inst.candidates = self._orig
        shutil.rmtree(self.d, ignore_errors=True)

    def test_skips_too_new_and_picks_supported(self):
        new = fake_python(self.d, "py314", "3.14")
        good = fake_python(self.d, "py312", "3.12")
        inst.candidates = lambda: [[str(new)], [str(good)]]
        cmd, rejected = inst.find_python(verbose=False)
        self.assertEqual(cmd, [str(good)])
        self.assertTrue(any("3.14" in r for r in rejected))

    def test_skips_lying_launcher_and_falls_through(self):
        liar = fake_python(self.d, "pyliar", "3.12", broken=True)
        good = fake_python(self.d, "py311", "3.11")
        inst.candidates = lambda: [[str(liar)], [str(good)]]
        cmd, _ = inst.find_python(verbose=False)
        self.assertEqual(cmd, [str(good)], "должен был пропустить врущий лаунчер")

    def test_nothing_suitable(self):
        only_new = fake_python(self.d, "py314", "3.14")
        inst.candidates = lambda: [[str(only_new)]]
        cmd, rejected = inst.find_python(verbose=False)
        self.assertIsNone(cmd)
        self.assertEqual(len(rejected), 1)
        self.assertIn("3.14", rejected[0])

    def test_rejected_list_has_no_duplicates(self):
        a = fake_python(self.d, "py314a", "3.14")
        b = fake_python(self.d, "py314b", "3.14")
        inst.candidates = lambda: [[str(a)], [str(b)]]
        _cmd, rejected = inst.find_python(verbose=False)
        self.assertEqual(len(rejected), 1)


class TestVenv(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self._root, self._venv, self._log = inst.ROOT, inst.VENV, inst.LOG
        inst.ROOT = self.d
        inst.VENV = self.d / ".venv"
        inst.LOG = self.d / "install_log.txt"

    def tearDown(self):
        inst.ROOT, inst.VENV, inst.LOG = self._root, self._venv, self._log
        shutil.rmtree(self.d, ignore_errors=True)

    def test_creates_and_verifies(self):
        self.assertTrue(inst.make_venv([sys.executable]))
        self.assertTrue(inst.venv_python().is_file())

    def test_detects_silent_failure(self):
        """Интерпретатор «отработал успешно», но окружения нет — должны поймать."""
        liar = fake_python(self.d, "pyliar", "3.12", exit_code=0, broken=True)
        self.assertFalse(inst.make_venv([str(liar)]),
                         "молчаливый провал venv должен быть замечен")

    def test_log_written(self):
        inst.make_venv([sys.executable])
        self.assertTrue(inst.LOG.is_file())
        self.assertIn("venv", inst.LOG.read_text(encoding="utf-8"))


class TestPipVerification(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self._log = inst.LOG
        inst.LOG = self.d / "log.txt"

    def tearDown(self):
        inst.LOG = self._log
        shutil.rmtree(self.d, ignore_errors=True)

    def test_import_check_catches_missing_module(self):
        """Даже если pip отчитался об успехе, отсутствие модуля должно всплыть."""
        ok = inst.pip_install(Path(sys.executable), [],
                              ["callqa_такого_модуля_нет"])
        self.assertFalse(ok)

    def test_import_check_passes_for_present_module(self):
        self.assertTrue(inst.pip_install(Path(sys.executable), [], ["json", "os"]))


class TestGroups(unittest.TestCase):
    def test_module_names_differ_from_package_names(self):
        """Частая ошибка: проверять импорт по имени пакета (PyYAML -> yaml)."""
        core = inst.GROUPS[0]
        self.assertIn("PyYAML", core[1])
        self.assertIn("yaml", core[2])
        self.assertIn("python-multipart", core[1])
        self.assertIn("multipart", core[2])

    def test_required_flags(self):
        titles = {g[0]: g[3] for g in inst.GROUPS}
        self.assertTrue(all(v for k, v in titles.items() if "речи" in k))
        self.assertFalse([g for g in inst.GROUPS if "Морфология" in g[0]][0][3],
                         "морфология должна быть необязательной")


if __name__ == "__main__":
    unittest.main(verbosity=2)
