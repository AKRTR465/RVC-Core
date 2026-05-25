import importlib
import unittest
from pathlib import Path

from tests.equivalence_helpers import fake_fairseq, make_temp_dir


class InferRuntimeEquivalenceTest(unittest.TestCase):
    def test_model_and_index_path_helpers_match_infer(self):
        with fake_fairseq():
            old_utils = importlib.import_module("infer.modules.vc.utils")
            new_utils = importlib.import_module("src.infer.model_utils")

            with make_temp_dir() as tmp:
                root = Path(tmp)
                export = root / "speaker" / "export"
                index = root / "speaker" / "index"
                export.mkdir(parents=True)
                index.mkdir(parents=True)
                (export / "voice.pth").write_bytes(b"model")
                (index / "voice.index").write_bytes(b"index")

                self.assertEqual(
                    old_utils.get_model_path_from_sid("voice", root),
                    new_utils.get_model_path_from_sid("voice", root),
                )
                self.assertEqual(
                    old_utils.get_index_path_from_model("voice", root),
                    new_utils.get_index_path_from_model("voice", root),
                )


if __name__ == "__main__":
    unittest.main()
