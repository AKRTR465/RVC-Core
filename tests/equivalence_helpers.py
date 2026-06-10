import contextlib
import os
import sys
import uuid
from pathlib import Path

import numpy as np
from scipy.io import wavfile


REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = Path(os.environ.get("RVC_REBUILD_TEST_TMP", REPO_ROOT / ".tmp_equivalence_tests"))


@contextlib.contextmanager
def patched_argv(argv):
    old_argv = sys.argv[:]
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old_argv


@contextlib.contextmanager
def make_temp_dir():
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        path = TMP_ROOT / f"tmp_{uuid.uuid4().hex}"
        try:
            path.mkdir()
            break
        except FileExistsError:
            continue
    yield str(path)


def write_sine_wav(path, sr=16000, seconds=2.0, freq=220.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wavfile.write(path, sr, audio)


def collect_binary_tree(root, ignored_names=None):
    ignored_names = set(ignored_names or ())
    root = Path(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ignored_names:
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def normalized_source(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return text.replace("\r\n", "\n").rstrip() + "\n"
