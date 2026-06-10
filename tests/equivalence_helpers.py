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
def fake_librosa():
    old_librosa = sys.modules.get("librosa")

    def resample(audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return np.asarray(audio).copy()
        old_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
        new_len = int(round(len(audio) * float(target_sr) / float(orig_sr)))
        new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
        return np.interp(new_x, old_x, audio).astype(np.asarray(audio).dtype)

    sys.modules["librosa"] = types.SimpleNamespace(resample=resample)
    try:
        yield
    finally:
        if old_librosa is None:
            sys.modules.pop("librosa", None)
        else:
            sys.modules["librosa"] = old_librosa


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
