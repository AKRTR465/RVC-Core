import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.utils import audio as audio_utils
from tests.equivalence_helpers import make_temp_dir


class AudioUtilsTest(unittest.TestCase):
    def test_load_audio_invokes_ffmpeg_cli_and_reads_float32_pcm(self):
        with make_temp_dir() as tmp:
            source = Path(tmp) / "input.wav"
            source.write_bytes(b"placeholder")
            expected = np.array([0.25, -0.5, 0.75], dtype=np.float32)

            with mock.patch(
                "src.utils.audio.subprocess.run",
                return_value=types.SimpleNamespace(stdout=expected.tobytes()),
            ) as run:
                actual = audio_utils.load_audio(source, 16000)

        np.testing.assert_array_equal(actual, expected)
        command = run.call_args.args[0]
        self.assertEqual(command[0:4], ["ffmpeg", "-nostdin", "-threads", "0"])
        self.assertIn("-acodec", command)
        self.assertIn("pcm_f32le", command)
        self.assertIn("-ac", command)
        self.assertIn("1", command)
        self.assertIn("-ar", command)
        self.assertIn("16000", command)

    def test_resample_audio_matches_librosa_default(self):
        librosa = self._optional_librosa()
        signal = np.sin(np.linspace(0, 12 * np.pi, 4097)).astype(np.float32)

        expected = librosa.resample(signal, orig_sr=48000, target_sr=16000)
        actual = audio_utils.resample_audio(signal, orig_sr=48000, target_sr=16000)

        np.testing.assert_array_equal(actual, expected)

    def test_audio_rms_matches_librosa_default(self):
        librosa = self._optional_librosa()
        signal = np.sin(np.linspace(0, 12 * np.pi, 4097)).astype(np.float32)

        expected = librosa.feature.rms(y=signal, frame_length=400, hop_length=160)
        actual = audio_utils.audio_rms(signal, frame_length=400, hop_length=160)

        np.testing.assert_array_equal(actual, expected)

    def _optional_librosa(self):
        try:
            import librosa
        except ImportError as exc:
            self.skipTest(f"librosa is not installed for equivalence testing: {exc}")
        return librosa
