"""Тесты доустановщика: подбор версии CUDA, распаковка ffmpeg, поиск ffmpeg в bin/."""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import install_extras as ex  # noqa: E402
from callqa import audio as au  # noqa: E402


def make_ffmpeg_zip(path: Path, layout: str = "nested") -> Path:
    """Собирает архив, похожий на настоящую сборку ffmpeg для Windows."""
    with zipfile.ZipFile(path, "w") as z:
        if layout == "nested":
            base = "ffmpeg-7.1-essentials_build/"
            z.writestr(base + "README.txt", "readme")
            z.writestr(base + "bin/ffmpeg.exe", b"MZ\x90\x00fake-ffmpeg")
            z.writestr(base + "bin/ffprobe.exe", b"MZ\x90\x00fake-ffprobe")
            z.writestr(base + "bin/ffplay.exe", b"MZ\x90\x00fake-ffplay")
            z.writestr(base + "doc/ffmpeg.html", "<html>не исполняемый</html>")
        elif layout == "flat":
            z.writestr("ffmpeg.exe", b"MZ\x90\x00fake")
            z.writestr("ffprobe.exe", b"MZ\x90\x00fake")
        elif layout == "empty":
            z.writestr("readme.txt", "тут ничего нет")
    return path


class TestCudaChannel(unittest.TestCase):
    def test_new_driver_gets_newest_cuda(self):
        self.assertEqual(ex.cuda_channel(560.0), "cu124")
        self.assertEqual(ex.cuda_channel(550.0), "cu124")

    def test_mid_driver(self):
        self.assertEqual(ex.cuda_channel(535.0), "cu121")
        self.assertEqual(ex.cuda_channel(525.0), "cu121")

    def test_old_driver(self):
        self.assertEqual(ex.cuda_channel(470.0), "cu118")

    def test_ancient_driver_gets_nothing(self):
        self.assertIsNone(ex.cuda_channel(390.0))
        self.assertIsNone(ex.cuda_channel(0.0))

    def test_thresholds_are_descending(self):
        needs = [n for n, _ in ex.CUDA_CHANNELS]
        self.assertEqual(needs, sorted(needs, reverse=True),
                         "пороги должны идти по убыванию, иначе выберется не тот канал")


class TestExtractFfmpeg(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_finds_exe_deep_inside_archive(self):
        z = make_ffmpeg_zip(self.d / "a.zip", "nested")
        out = ex.extract_ffmpeg(z, self.d / "bin")
        self.assertEqual(out, {"ffmpeg.exe", "ffprobe.exe"})
        self.assertTrue((self.d / "bin" / "ffmpeg.exe").is_file())
        self.assertTrue((self.d / "bin" / "ffprobe.exe").is_file())

    def test_ignores_unrelated_files(self):
        z = make_ffmpeg_zip(self.d / "a.zip", "nested")
        ex.extract_ffmpeg(z, self.d / "bin")
        got = {p.name for p in (self.d / "bin").iterdir()}
        self.assertNotIn("ffplay.exe", got)
        self.assertNotIn("README.txt", got)
        self.assertNotIn("ffmpeg.html", got)

    def test_flat_archive(self):
        z = make_ffmpeg_zip(self.d / "a.zip", "flat")
        self.assertEqual(ex.extract_ffmpeg(z, self.d / "bin"),
                         {"ffmpeg.exe", "ffprobe.exe"})

    def test_archive_without_ffmpeg(self):
        z = make_ffmpeg_zip(self.d / "a.zip", "empty")
        self.assertEqual(ex.extract_ffmpeg(z, self.d / "bin"), set())

    def test_creates_destination(self):
        z = make_ffmpeg_zip(self.d / "a.zip", "flat")
        target = self.d / "нет" / "такой" / "папки"
        ex.extract_ffmpeg(z, target)
        self.assertTrue((target / "ffmpeg.exe").is_file())

    def test_no_zip_slip(self):
        """Путь из архива не должен уводить запись за пределы папки."""
        z = self.d / "evil.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("../../../ffmpeg.exe", b"MZ")
        ex.extract_ffmpeg(z, self.d / "bin")
        self.assertTrue((self.d / "bin" / "ffmpeg.exe").is_file())
        self.assertFalse((self.d.parent / "ffmpeg.exe").exists())

    def test_bad_zip_raises(self):
        bad = self.d / "bad.zip"
        bad.write_bytes("это не архив".encode("utf-8"))
        with self.assertRaises(zipfile.BadZipFile):
            ex.extract_ffmpeg(bad, self.d / "bin")


class TestLocalBinDiscovery(unittest.TestCase):
    """ffmpeg, положенный в bin/ проекта, должен находиться без правки PATH."""

    def setUp(self):
        self.bin = ROOT / "bin"
        self.created = not self.bin.exists()
        self.bin.mkdir(exist_ok=True)
        self.name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        self.fake = self.bin / self.name
        self.existed = self.fake.exists()
        if not self.existed:
            self.fake.write_bytes(b"fake")

    def tearDown(self):
        if not self.existed and self.fake.exists():
            self.fake.unlink()
        if self.created and self.bin.exists() and not any(self.bin.iterdir()):
            self.bin.rmdir()

    def test_found_in_project_bin(self):
        found = au.ffmpeg_path()
        self.assertIsNotNone(found)
        self.assertEqual(Path(found).parent.resolve(), self.bin.resolve(),
                         "должен предпочитать bin/ проекта")

    def test_bin_added_to_path(self):
        """Папка bin/ должна попадать в PATH — ffmpeg вызывают и сторонние библиотеки."""
        import callqa

        callqa._add_local_tools_to_path()
        self.assertIn(str(self.bin.resolve()),
                      [str(Path(p).resolve()) for p in
                       os.environ.get("PATH", "").split(os.pathsep) if p])

    def test_path_not_duplicated(self):
        import callqa

        callqa._add_local_tools_to_path()
        callqa._add_local_tools_to_path()
        target = str(self.bin.resolve())
        count = sum(1 for p in os.environ.get("PATH", "").split(os.pathsep)
                    if p and str(Path(p).resolve()) == target)
        self.assertEqual(count, 1)


class TestGpuDetection(unittest.TestCase):
    def test_returns_expected_shape(self):
        info = ex.detect_gpu()
        self.assertIn("present", info)
        self.assertIn("name", info)
        self.assertIn("driver", info)
        self.assertIsInstance(info["present"], bool)

    def test_no_crash_without_nvidia(self):
        info = ex.detect_gpu()
        if not info["present"]:
            self.assertEqual(info["name"], "")


class TestStatus(unittest.TestCase):
    def test_status_keys(self):
        st = ex.status()
        for key in ("ffmpeg", "gpu_hw", "cuda_asr", "local_llm",
                    "api_key", "torch", "pyannote"):
            self.assertIn(key, st)

    def test_print_status_runs(self):
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            ex.print_status(ex.status())
        finally:
            sys.stdout = old
        text = buf.getvalue()
        for expected in ("ffmpeg", "pyannote", "локальная модель",
                         "CUDA для распознавания"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
