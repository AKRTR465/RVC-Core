import ast
import unittest

from tests.equivalence_helpers import REPO_ROOT, normalized_source


ENTRYPOINTS = {
    "src/__init__.py",
    "src/index/__init__.py",
    "src/index/__main__.py",
    "src/infer/__init__.py",
    "src/preprocess/__init__.py",
    "src/preprocess/__main__.py",
    "src/train/__init__.py",
    "src/train/__main__.py",
    "src/utils/__init__.py",
    "src/utils/infer_pack/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
}

COMPATIBILITY_WRAPPERS = {
    "src/train/process_ckpt.py": "src.train.checkpoint_export",
    "src/utils/infer_pack/attentions.py": "src.models.attentions",
    "src/utils/infer_pack/commons.py": "src.models.commons",
    "src/utils/infer_pack/models.py": "src.models.models",
    "src/utils/infer_pack/modules.py": "src.models.modules",
    "src/utils/infer_pack/transforms.py": "src.models.transforms",
}

CANONICAL_EXPORT_MODULES = {
    "src/models/attentions.py",
    "src/models/commons.py",
    "src/models/models.py",
    "src/models/modules.py",
    "src/models/transforms.py",
}

CORE_MODULES = {
    "src/features/f0.py",
    "src/features/hubert.py",
    "src/features/mel.py",
    "src/index/build_v1.py",
    "src/index/build_v2.py",
    "src/index/builder.py",
    "src/index/common.py",
    "src/index/retrieval.py",
    "src/infer/batch.py",
    "src/infer/model_utils.py",
    "src/infer/pipeline.py",
    "src/infer/service.py",
    "src/infer/voice_converter.py",
    "src/preprocess/audio.py",
    "src/preprocess/f0.py",
    "src/preprocess/features.py",
    "src/preprocess/pipeline.py",
    "src/train/checkpoint_export.py",
    "src/train/data_utils.py",
    "src/train/losses.py",
    "src/train/mel_processing.py",
    "src/train/runner.py",
    "src/train/utils.py",
    "src/utils/audio.py",
    "src/utils/rmvpe.py",
}

EXPECTED_SRC_FILES = (
    ENTRYPOINTS
    | set(COMPATIBILITY_WRAPPERS)
    | CANONICAL_EXPORT_MODULES
    | CORE_MODULES
)


def assigned_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


class SourceCoverageEquivalenceTest(unittest.TestCase):
    def test_legacy_infer_tree_is_removed(self):
        self.assertFalse((REPO_ROOT / "infer").exists())

    def test_every_src_python_file_is_classified(self):
        src_files = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "src").rglob("*.py")
            if "__pycache__" not in path.parts
        }
        self.assertEqual(src_files, EXPECTED_SRC_FILES)

    def test_compatibility_wrappers_only_reexport_canonical_modules(self):
        for wrapper_rel, canonical_module in COMPATIBILITY_WRAPPERS.items():
            text = normalized_source(REPO_ROOT / wrapper_rel)
            if wrapper_rel == "src/train/process_ckpt.py":
                self.assertIn(f"from {canonical_module} import", text)
                self.assertIn("__all__", text)
                continue
            self.assertEqual(
                text,
                f"from {canonical_module} import *  # noqa: F401,F403\n",
                wrapper_rel,
            )

    def test_canonical_wildcard_targets_define_exports(self):
        for rel_path in CANONICAL_EXPORT_MODULES:
            tree = ast.parse(normalized_source(REPO_ROOT / rel_path), filename=rel_path)
            self.assertIn("__all__", assigned_names(tree), rel_path)

    def test_src_has_no_old_infer_imports(self):
        offenders = []
        for path in (REPO_ROOT / "src").rglob("*.py"):
            text = normalized_source(path)
            if "from infer" in text or "import infer" in text or "infer.lib" in text:
                offenders.append(path.relative_to(REPO_ROOT).as_posix())
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
