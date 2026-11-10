"""CallQA — анализ записей телефонных разговоров.

Вход: аудиофайл (или папка). Выход: таблица «что было / чего не было»
по заданному чек-листу, метрики диалога и балл качества.
"""
__version__ = "1.1.0"

import os as _os


def _add_local_tools_to_path() -> None:
    """Кладёт папку bin/ проекта в PATH.

    Туда install_extras.py распаковывает ffmpeg — так он доступен программе,
    не требуя прав администратора и правки системных переменных.
    """
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    local_bin = _os.path.join(root, "bin")
    if not _os.path.isdir(local_bin):
        return
    current = _os.environ.get("PATH", "")
    parts = current.split(_os.pathsep)
    if local_bin not in parts:
        _os.environ["PATH"] = local_bin + _os.pathsep + current


_add_local_tools_to_path()

from .config import Config  # noqa: E402,F401
from .models import CallResult, Transcript  # noqa: E402,F401

__all__ = ["Config", "CallResult", "Transcript", "__version__"]
