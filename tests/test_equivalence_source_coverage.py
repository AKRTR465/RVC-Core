import unittest
from pathlib import Path

from tests.equivalence_helpers import REPO_ROOT, normalized_source


DIRECT_EQUIVALENCE = {
    "infer/lib/audio.py": "src/utils/audio.py",
    "infer/lib/rmvpe.py": "src/utils/rmvpe.py",
    "infer/lib/slicer2.py": "src/preprocess/utils/slicer.py",
    "infer/lib/infer_pack/attentions.py": "src/utils/infer_pack/attentions.py",
    "infer/lib/infer_pack/commons.py": "src/utils/infer_pack/commons.py",
    "infer/lib/infer_pack/models.py": "src/utils/infer_pack/models.py",
    "infer/lib/infer_pack/modules.py": "src/utils/infer_pack/modules.py",
    "infer/lib/infer_pack/transforms.py": "src/utils/infer_pack/transforms.py",
    "infer/lib/train/data_utils.py": "src/train/data_utils.py",
    "infer/lib/train/losses.py": "src/train/losses.py",
    "infer/lib/train/mel_processing.py": "src/train/mel_processing.py",
    "infer/lib/train/process_ckpt.py": "src/train/process_ckpt.py",
    "infer/modules/vc/utils.py": "src/infer/model_utils.py",
}

REFACTORED_EQUIVALENCE = {
    "infer/index/train-index.py": ["src/index/build_v1.py", "src/index/common.py"],
    "infer/index/train-index-v2.py": ["src/index/build_v2.py", "src/index/common.py"],
    "infer/modules/train/preprocess.py": ["src/preprocess/audio.py"],
    "infer/modules/train/extract_f0_print.py": ["src/preprocess/f0.py"],
    "infer/modules/train/extract_feature_print.py": ["src/preprocess/features.py"],
    "infer/modules/train/train.py": ["src/train/runner.py"],
    "infer/lib/train/utils.py": ["src/train/utils.py"],
    "infer/modules/vc/__init__.py": ["src/infer/__init__.py"],
    "infer/modules/vc/modules.py": ["src/infer/voice_converter.py"],
    "infer/modules/vc/pipeline.py": ["src/infer/pipeline.py"],
}

NEW_ENTRYPOINTS = {
    "src/__init__.py",
    "src/index/__init__.py",
    "src/index/__main__.py",
    "src/infer/__init__.py",
    "src/preprocess/__init__.py",
    "src/preprocess/__main__.py",
    "src/preprocess/utils/__init__.py",
    "src/train/__init__.py",
    "src/train/__main__.py",
    "src/utils/__init__.py",
    "src/utils/infer_pack/__init__.py",
}


class SourceCoverageEquivalenceTest(unittest.TestCase):
    def test_every_infer_python_file_has_src_equivalence(self):
        covered = set(DIRECT_EQUIVALENCE) | set(REFACTORED_EQUIVALENCE)
        infer_files = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "infer").rglob("*.py")
            if "__pycache__" not in path.parts
        }
        self.assertEqual(infer_files, covered)

    def test_every_src_python_file_is_classified(self):
        classified = set(DIRECT_EQUIVALENCE.values()) | NEW_ENTRYPOINTS
        for values in REFACTORED_EQUIVALENCE.values():
            classified.update(values)
        src_files = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "src").rglob("*.py")
            if "__pycache__" not in path.parts
        }
        self.assertEqual(src_files, classified)

    def test_direct_copy_modules_only_change_import_roots(self):
        replacements = {
            "infer.lib.audio": "src.utils.audio",
            "infer.lib.rmvpe": "src.utils.rmvpe",
            "infer.lib.slicer2": "src.preprocess.utils.slicer",
            "infer.lib.infer_pack": "src.utils.infer_pack",
            "infer.lib.train": "src.train",
            "infer.modules.vc.utils": "src.infer.model_utils",
        }
        for old_rel, new_rel in DIRECT_EQUIVALENCE.items():
            old_text = normalized_source(REPO_ROOT / old_rel)
            for old, new in replacements.items():
                old_text = old_text.replace(old, new)
            new_text = normalized_source(REPO_ROOT / new_rel)
            self.assertEqual(new_text, old_text, f"{new_rel} drifted from {old_rel}")

    def test_src_has_no_old_infer_imports(self):
        offenders = []
        for path in (REPO_ROOT / "src").rglob("*.py"):
            text = normalized_source(path)
            if "from infer" in text or "import infer" in text or "infer.lib" in text:
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
