import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

PREPROCESS_ITEMS = {
    "0_gt_wavs",
    "1_16k_wavs",
    "2a_f0",
    "2b-f0nsf",
    "3_feature256",
    "3_feature768",
    "filelist.txt",
    "preprocess.log",
    "extract_f0_feature.log",
}

TRAIN_ITEMS = {"eval", "train.log", "config.json", "githash"}


def guess_project_name(stem: str) -> str:
    if "_e" in stem and "_s" in stem:
        return stem.split("_e", 1)[0]
    return stem


def move_path(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        logger.warning("skip existing target: %s -> %s", src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    logger.info("move %s -> %s", src, dst)
    if not dry_run:
        shutil.move(str(src), str(dst))


def migrate_pretrain(repo_root: Path, dry_run: bool) -> None:
    assets_root = repo_root / "assets"
    pretrain_root = repo_root / "pretrain"
    for name in ("hubert", "rmvpe", "pretrained", "pretrained_v2"):
        src_dir = assets_root / name
        dst_dir = pretrain_root / name
        if not src_dir.exists():
            continue
        for src in src_dir.iterdir():
            if src.name == ".gitignore":
                continue
            move_path(src, dst_dir / src.name, dry_run)


def migrate_asset_outputs(repo_root: Path, dry_run: bool) -> None:
    weights_root = repo_root / "assets" / "weights"
    indices_root = repo_root / "assets" / "indices"
    ckpt_root = repo_root / "ckpt"

    if weights_root.exists():
        for src in weights_root.iterdir():
            if src.name == ".gitignore":
                continue
            project_name = guess_project_name(src.stem)
            move_path(src, ckpt_root / project_name / "export" / src.name, dry_run)

    if indices_root.exists():
        for src in indices_root.iterdir():
            if src.name == ".gitignore":
                continue
            project_name = guess_project_name(src.stem)
            move_path(src, ckpt_root / project_name / "index" / src.name, dry_run)


def migrate_logs(repo_root: Path, dry_run: bool) -> None:
    logs_root = repo_root / "logs"
    data_root = repo_root / "data"
    ckpt_root = repo_root / "ckpt"
    if not logs_root.exists():
        return

    for exp_dir in logs_root.iterdir():
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        preprocess_root = data_root / name / "preprocess_data"
        work_dir = ckpt_root / name
        train_root = ckpt_root / name / "train"
        export_root = ckpt_root / name / "export"
        index_dir = ckpt_root / name / "index"

        for src in exp_dir.iterdir():
            if src.name in PREPROCESS_ITEMS:
                move_path(src, preprocess_root / src.name, dry_run)
                continue
            if src.name in {"config.yaml", "resolved_config.yaml"}:
                move_path(src, work_dir / "config.yaml", dry_run)
                continue
            if src.name in TRAIN_ITEMS or src.name.startswith(("G_", "D_")):
                move_path(src, train_root / src.name, dry_run)
                continue
            if src.suffix == ".index":
                move_path(src, index_dir / src.name, dry_run)
                continue
            if src.suffix == ".pth":
                move_path(src, export_root / src.name, dry_run)
                continue
            move_path(src, train_root / src.name, dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    repo_root = Path(__file__).resolve().parent.parent

    migrate_pretrain(repo_root, args.dry_run)
    migrate_asset_outputs(repo_root, args.dry_run)
    migrate_logs(repo_root, args.dry_run)


if __name__ == "__main__":
    main()
