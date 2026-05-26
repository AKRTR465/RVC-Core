import unittest

import numpy as np

try:
    from src.train import utils
except ModuleNotFoundError as exc:
    missing_dependency = exc.name
    utils = None
else:
    missing_dependency = None


def install_tostring_rgb_compat():
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if hasattr(FigureCanvasAgg, "tostring_rgb"):
        return

    def tostring_rgb(self):
        self.draw()
        return np.asarray(self.buffer_rgba())[:, :, :3].tobytes()

    FigureCanvasAgg.tostring_rgb = tostring_rgb


@unittest.skipIf(utils is None, f"missing dependency: {missing_dependency}")
class VisualOutputTest(unittest.TestCase):
    def assert_rgb_image(self, image):
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(image.shape[0], 0)
        self.assertGreater(image.shape[1], 0)

    def test_spectrogram_image_output_is_stable(self):
        install_tostring_rgb_compat()
        spectrogram = np.linspace(-1.0, 1.0, 80 * 120, dtype=np.float32).reshape(80, 120)
        first = utils.plot_spectrogram_to_numpy(spectrogram)
        second = utils.plot_spectrogram_to_numpy(spectrogram)
        self.assert_rgb_image(first)
        np.testing.assert_array_equal(first, second)

    def test_alignment_image_output_is_stable(self):
        install_tostring_rgb_compat()
        alignment = np.linspace(0.0, 1.0, 32 * 48, dtype=np.float32).reshape(32, 48)
        first = utils.plot_alignment_to_numpy(alignment, info="coverage")
        second = utils.plot_alignment_to_numpy(alignment, info="coverage")
        self.assert_rgb_image(first)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
