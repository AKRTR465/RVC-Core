import importlib
import unittest

import numpy as np


def install_tostring_rgb_compat():
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if hasattr(FigureCanvasAgg, "tostring_rgb"):
        return

    def tostring_rgb(self):
        self.draw()
        return np.asarray(self.buffer_rgba())[:, :, :3].tobytes()

    FigureCanvasAgg.tostring_rgb = tostring_rgb


class VisualEquivalenceTest(unittest.TestCase):
    def test_spectrogram_image_output_matches_infer(self):
        install_tostring_rgb_compat()
        old_utils = importlib.import_module("infer.lib.train.utils")
        new_utils = importlib.import_module("src.train.utils")

        spectrogram = np.linspace(-1.0, 1.0, 80 * 120, dtype=np.float32).reshape(80, 120)
        np.testing.assert_array_equal(
            old_utils.plot_spectrogram_to_numpy(spectrogram),
            new_utils.plot_spectrogram_to_numpy(spectrogram),
        )

    def test_alignment_image_output_matches_infer(self):
        install_tostring_rgb_compat()
        old_utils = importlib.import_module("infer.lib.train.utils")
        new_utils = importlib.import_module("src.train.utils")

        alignment = np.linspace(0.0, 1.0, 32 * 48, dtype=np.float32).reshape(32, 48)
        np.testing.assert_array_equal(
            old_utils.plot_alignment_to_numpy(alignment, info="equivalence"),
            new_utils.plot_alignment_to_numpy(alignment, info="equivalence"),
        )


if __name__ == "__main__":
    unittest.main()
