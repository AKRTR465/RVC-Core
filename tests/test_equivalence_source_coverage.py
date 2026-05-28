import ast
import unittest

from tests.equivalence_helpers import REPO_ROOT, normalized_source


CANONICAL_EXPORT_MODULES = {
    "src/models/attentions.py",
    "src/models/commons.py",
    "src/models/models.py",
    "src/models/modules.py",
    "src/models/transforms.py",
}


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
        self.assertFalse((REPO_ROOT / "src/infer/batch.py").exists())
        self.assertFalse((REPO_ROOT / "src/infer/service.py").exists())
        self.assertFalse((REPO_ROOT / "src/infer/voice_converter.py").exists())
        self.assertFalse((REPO_ROOT / "src/train/process_ckpt.py").exists())
        self.assertFalse((REPO_ROOT / "src/index/build_v1.py").exists())
        self.assertFalse((REPO_ROOT / "src/index/build_v2.py").exists())
        self.assertFalse((REPO_ROOT / "src/utils/infer_pack/__init__.py").exists())
        self.assertFalse((REPO_ROOT / "src/utils/infer_pack/attentions.py").exists())
        self.assertFalse((REPO_ROOT / "src/utils/infer_pack/commons.py").exists())
        self.assertFalse((REPO_ROOT / "src/utils/infer_pack/models.py").exists())
        self.assertFalse((REPO_ROOT / "src/utils/infer_pack/modules.py").exists())
        self.assertFalse((REPO_ROOT / "src/utils/infer_pack/transforms.py").exists())

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
