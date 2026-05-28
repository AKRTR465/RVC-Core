from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from configs.project_config import load_project_config, parse_hparams_overrides
from src.features.f0 import F0_METHODS
from src.preprocess import audio as audio_stage
from src.preprocess import f0 as f0_stage
from src.preprocess import features as feature_stage
from src.preprocess.common import log_message, run_worker_shards
from src.preprocess.layout import MANIFEST_NAME, PreprocessLayout
from src.rvc_profiles import get_feature_profile

DEFAULT_STAGES = ("audio", "f0", "features", "filelist")
VALID_STAGES = set(DEFAULT_STAGES)
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
}


@dataclass(frozen=True)
class DatasetItem:
    source_path: Path
    speaker_id: int
    index: int


def _layout_for_project(project: dict) -> PreprocessLayout:
    return PreprocessLayout(
        root=Path(project["paths"]["preprocess_dir"]),
        feature_profile=get_feature_profile(str(project["selectors"]["version"])),
    )


def parse_stage_list(raw_stages: str | None) -> tuple[str, ...]:
    if raw_stages in (None, ""):
        return DEFAULT_STAGES

    stages = tuple(stage.strip().lower() for stage in raw_stages.split(",") if stage.strip())
    if not stages:
        raise ValueError("--stages cannot be empty")

    invalid = sorted(set(stages) - VALID_STAGES)
    if invalid:
        raise ValueError(
            f"Unsupported stage(s): {', '.join(invalid)}. Expected any of: {', '.join(DEFAULT_STAGES)}"
        )
    return tuple(dict.fromkeys(stages))


def _is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def _direct_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if _is_audio_file(path))


def _visible_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))


def _speaker_embed_dim(project: dict) -> int:
    value = project.get("model", {}).get("spk_embed_dim", 1)
    try:
        spk_embed_dim = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model.spk_embed_dim must be an integer, got: {value!r}") from exc
    if spk_embed_dim < 1:
        raise ValueError(f"model.spk_embed_dim must be >= 1, got: {spk_embed_dim}")
    return spk_embed_dim


def discover_dataset_items(dataset_dir: str | Path, spk_embed_dim: int) -> list[DatasetItem]:
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist or is not a directory: {root}")

    top_level_files = _direct_audio_files(root)
    speaker_dirs = _visible_dirs(root)
    if top_level_files and speaker_dirs:
        raise ValueError("dataset_dir cannot mix top-level audio files and speaker subdirectories")

    if top_level_files:
        return [
            DatasetItem(source_path=path.resolve(), speaker_id=0, index=index)
            for index, path in enumerate(top_level_files)
        ]

    if not speaker_dirs:
        raise ValueError(f"No audio files found under dataset_dir: {root}")

    numbered_dirs: list[tuple[int, Path]] = []
    for speaker_dir in speaker_dirs:
        if not speaker_dir.name.isdigit() or int(speaker_dir.name) < 1:
            raise ValueError(
                "Multi-speaker dataset directories must be positive integers, "
                f"got: {speaker_dir.name!r}"
            )
        numbered_dirs.append((int(speaker_dir.name), speaker_dir))

    max_speaker_id = max(speaker_id for speaker_id, _ in numbered_dirs)
    if max_speaker_id > spk_embed_dim:
        raise ValueError(
            f"Max speaker directory id {max_speaker_id} exceeds model.spk_embed_dim={spk_embed_dim}"
        )

    items: list[DatasetItem] = []
    for speaker_dir_id, speaker_dir in sorted(numbered_dirs):
        sid = speaker_dir_id - 1
        for source_path in _direct_audio_files(speaker_dir):
            items.append(
                DatasetItem(
                    source_path=source_path.resolve(),
                    speaker_id=sid,
                    index=len(items),
                )
            )
    if not items:
        raise ValueError(f"No audio files found under speaker directories in: {root}")
    return items


def _output_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem.rsplit("_", 1)[1]), path.name
    except (IndexError, ValueError):
        return 0, path.name


def _manifest_records(items: list[DatasetItem], layout: PreprocessLayout) -> list[dict]:
    records: list[dict] = []
    for item in items:
        for gt_wav in sorted(layout.gt_wavs_dir.glob(f"{item.index}_*.wav"), key=_output_sort_key):
            wav16k = layout.wav16k_dir / gt_wav.name
            if not wav16k.is_file():
                continue
            records.append(
                {
                    "source_path": str(item.source_path),
                    "speaker_id": item.speaker_id,
                    "gt_wav": str(gt_wav.resolve()),
                    "wav16k": str(wav16k.resolve()),
                }
            )
    return records


def write_preprocess_manifest(items: list[DatasetItem], preprocess_dir: str | Path) -> Path:
    layout = PreprocessLayout(Path(preprocess_dir), get_feature_profile("v1"))
    layout.root.mkdir(parents=True, exist_ok=True)
    records = _manifest_records(items, layout)
    if not records:
        raise RuntimeError("Audio preprocessing produced no manifest rows")

    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    layout.manifest_path.write_text(f"{payload}\n", encoding="utf-8")
    log_message(layout.preprocess_log_path, f"wrote manifest rows={len(records)}: {layout.manifest_path}")
    return layout.manifest_path


def _write_project_manifest(items: list[DatasetItem], layout: PreprocessLayout) -> Path:
    layout.root.mkdir(parents=True, exist_ok=True)
    records = _manifest_records(items, layout)
    if not records:
        raise RuntimeError("Audio preprocessing produced no manifest rows")

    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    layout.manifest_path.write_text(f"{payload}\n", encoding="utf-8")
    log_message(layout.preprocess_log_path, f"wrote manifest rows={len(records)}: {layout.manifest_path}")
    return layout.manifest_path


def load_preprocess_manifest(preprocess_dir: str | Path) -> list[dict]:
    manifest_path = Path(preprocess_dir) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run the audio stage before generating filelist."
        )

    records: list[dict] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"source_path", "speaker_id", "gt_wav", "wav16k"} - set(record)
            if missing:
                raise ValueError(
                    f"Invalid manifest row {line_number}: missing {', '.join(sorted(missing))}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


def _if_f0(project: dict) -> int:
    return int(project["selectors"]["if_f0"])


def _build_filelist_rows(project: dict, records: list[dict], layout: PreprocessLayout) -> tuple[list[str], int]:
    use_f0 = _if_f0(project) == 1
    rows: list[str] = []
    skipped = 0
    for record in records:
        gt_wav = Path(record["gt_wav"])
        wav16k = Path(record["wav16k"])
        feature_path = layout.feature_dir / wav16k.with_suffix(".npy").name
        sid = int(record["speaker_id"])

        required_paths = [gt_wav, feature_path]
        if use_f0:
            coarse_f0 = layout.f0_dir / f"{wav16k.name}.npy"
            nsf_f0 = layout.f0nsf_dir / f"{wav16k.name}.npy"
            required_paths.extend([coarse_f0, nsf_f0])

        if not all(path.is_file() for path in required_paths):
            skipped += 1
            continue

        if use_f0:
            rows.append("|".join(map(str, [gt_wav, feature_path, coarse_f0, nsf_f0, sid])))
        else:
            rows.append("|".join(map(str, [gt_wav, feature_path, sid])))

    return rows, skipped


def _resolve_validation_split(project: dict) -> float:
    value = project.get("preprocess", {}).get("validation_split", 0.1)
    try:
        validation_split = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"preprocess.validation_split must be a float between 0 and 1, got: {value!r}"
        ) from exc
    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            "preprocess.validation_split must be > 0 and < 1 "
            f"to generate a required validation set, got: {validation_split}"
        )
    return validation_split


def _resolve_validation_seed(project: dict) -> int:
    value = project.get("preprocess", {}).get("validation_seed", 1234)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"preprocess.validation_seed must be an integer, got: {value!r}") from exc


def _speaker_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def split_filelist_rows(
    rows: list[str],
    validation_split: float,
    validation_seed: int,
) -> tuple[list[str], list[str]]:
    speaker_groups: dict[str, list[str]] = {}
    for row in rows:
        sid = row.split("|")[-1]
        speaker_groups.setdefault(sid, []).append(row)

    rng = random.Random(int(validation_seed))
    grouped_train_lines: dict[str, list[str]] = {}
    grouped_val_lines: dict[str, list[str]] = {}

    for sid in sorted(speaker_groups.keys(), key=_speaker_sort_key):
        group = list(speaker_groups[sid])
        rng.shuffle(group)
        if len(group) <= 1:
            grouped_train_lines[sid] = group
            grouped_val_lines[sid] = []
            continue
        n_val = int(len(group) * float(validation_split))
        n_val = max(0, min(n_val, len(group) - 1))
        grouped_val_lines[sid] = group[:n_val]
        grouped_train_lines[sid] = group[n_val:]

    val_lines = []
    for sid in sorted(grouped_val_lines.keys(), key=_speaker_sort_key):
        val_lines.extend(grouped_val_lines[sid])

    if not val_lines:
        donor_sids = [
            sid
            for sid in sorted(speaker_groups.keys(), key=_speaker_sort_key)
            if len(speaker_groups[sid]) > 1 and grouped_train_lines.get(sid)
        ]
        if donor_sids:
            best_sid = max(
                donor_sids,
                key=lambda sid: (len(speaker_groups[sid]), _speaker_sort_key(sid)),
            )
            grouped_val_lines[best_sid].append(grouped_train_lines[best_sid].pop(0))
        else:
            raise RuntimeError(
                "Validation split produced no validation samples. Add more data or adjust preprocess.validation_split."
            )

    train_lines: list[str] = []
    val_lines = []
    for sid in sorted(speaker_groups.keys(), key=_speaker_sort_key):
        train_lines.extend(grouped_train_lines.get(sid, []))
        val_lines.extend(grouped_val_lines.get(sid, []))

    return train_lines, val_lines


def generate_filelist(project: dict) -> tuple[Path, int, int]:
    layout = _layout_for_project(project)
    records = load_preprocess_manifest(layout.root)
    rows, skipped = _build_filelist_rows(project, records, layout)
    if not rows:
        raise RuntimeError(
            "No valid preprocess samples found for filelist. "
            f"manifest_rows={len(records)}, skipped={skipped}"
        )

    layout.filelist_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    log_message(layout.preprocess_log_path, f"wrote filelist rows={len(rows)}, skipped={skipped}: {layout.filelist_path}")

    train_lines, val_lines = split_filelist_rows(
        rows,
        _resolve_validation_split(project),
        _resolve_validation_seed(project),
    )
    layout.train_filelist_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    layout.val_filelist_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    log_message(layout.preprocess_log_path, f"wrote train filelist rows={len(train_lines)}: {layout.train_filelist_path}")
    log_message(layout.preprocess_log_path, f"wrote val filelist rows={len(val_lines)}: {layout.val_filelist_path}")
    return layout.filelist_path, len(rows), skipped


def _resolve_audio_workers(project: dict, workers_override: int | None) -> int:
    runtime = project["runtime"]
    return int(workers_override if workers_override is not None else runtime["n_cpu"])


def _resolve_f0_workers(project: dict, f0method: str, workers_override: int | None) -> int:
    if workers_override is not None:
        return int(workers_override)
    return 1 if f0method == "rmvpe" else int(project["runtime"]["n_cpu"])


def _resolve_runtime_device(project: dict, device_override: str | None) -> str:
    if device_override not in {None, "", "auto"}:
        return str(device_override)
    return str(project["runtime"]["device"])


def _resolve_runtime_is_half(project: dict, is_half_override: bool | None) -> bool:
    if is_half_override is not None:
        return bool(is_half_override)
    return bool(project["runtime"]["is_half"])


def run_audio_stage(project: dict, workers_override: int | None = None) -> Path:
    layout = _layout_for_project(project)
    layout.root.mkdir(parents=True, exist_ok=True)

    items = discover_dataset_items(project["paths"]["dataset_dir"], _speaker_embed_dim(project))
    sampling_rate = int(project["data"]["sampling_rate"])
    preprocess_config = project.get("preprocess", {})
    workers = min(_resolve_audio_workers(project, workers_override), len(items))
    workers = max(workers, 1)
    noparallel = bool(preprocess_config.get("noparallel", False))

    log_message(
        layout.preprocess_log_path,
        f"start audio preprocess, items={len(items)}, workers={workers}, noparallel={noparallel}, mode=prepared-audio",
    )
    payload = [(str(item.source_path), item.index) for item in items]
    run_worker_shards(
        payload,
        workers,
        audio_stage.run_audio_items,
        lambda shard: (shard, sampling_rate, layout.root),
        error_label="audio preprocess worker",
        parallel=not noparallel,
    )
    manifest_path = _write_project_manifest(items, layout)
    log_message(layout.preprocess_log_path, "end audio preprocess")
    return manifest_path


def run_f0_stage(
    project: dict,
    f0method: str,
    workers_override: int | None = None,
    *,
    device_override: str | None = None,
    is_half_override: bool | None = None,
) -> None:
    if _if_f0(project) == 0:
        log_message(_layout_for_project(project).feature_log_path, "skip f0 stage because selectors.if_f0=0")
        return
    if f0method not in F0_METHODS:
        raise ValueError(f"Unsupported f0method={f0method!r}")

    layout = _layout_for_project(project)
    layout.root.mkdir(parents=True, exist_ok=True)
    workers = _resolve_f0_workers(project, f0method, workers_override)
    device_request = _resolve_runtime_device(project, device_override)
    is_half_request = _resolve_runtime_is_half(project, is_half_override)
    device, is_half = f0_stage.resolve_runtime(
        f0method,
        device_request,
        is_half_request,
        layout.feature_log_path,
    )
    pretrain_root = str(project["paths"]["pretrain_root"])
    paths = f0_stage.build_paths(layout)

    log_message(
        layout.feature_log_path,
        f"pipeline-f0,method={f0method},workers={workers},device={device},is_half={is_half}",
    )
    run_worker_shards(
        paths,
        workers,
        f0_stage.run_worker,
        lambda shard: (shard, f0method, device, is_half, pretrain_root, layout.feature_log_path),
        error_label="f0 worker",
    )


def run_feature_stage(
    project: dict,
    *,
    device_override: str | None = None,
    is_half_override: bool | None = None,
) -> None:
    layout = _layout_for_project(project)
    layout.root.mkdir(parents=True, exist_ok=True)
    device = feature_stage.resolve_device(_resolve_runtime_device(project, device_override))
    feature_stage.extract_features(
        layout=layout,
        n_part=1,
        i_part=0,
        device=device,
        is_half=_resolve_runtime_is_half(project, is_half_override),
        model_path=str(project["paths"]["hubert_path"]),
    )


def run_pipeline(
    project: dict,
    stages: tuple[str, ...],
    f0method: str,
    workers_override: int | None = None,
    *,
    device_override: str | None = None,
    is_half_override: bool | None = None,
) -> None:
    for stage in stages:
        if stage == "audio":
            run_audio_stage(project, workers_override)
        elif stage == "f0":
            run_f0_stage(
                project,
                f0method,
                workers_override,
                device_override=device_override,
                is_half_override=is_half_override,
            )
        elif stage == "features":
            run_feature_stage(
                project,
                device_override=device_override,
                is_half_override=is_half_override,
            )
        elif stage == "filelist":
            generate_filelist(project)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported stage: {stage}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the unified RVC preprocessing pipeline."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--stages",
        type=str,
        default="",
        help="Comma-separated subset for repair/debug runs. Default: audio,f0,features,filelist",
    )
    parser.add_argument(
        "--f0method",
        type=str,
        choices=sorted(F0_METHODS),
        default="",
        help="Override preprocess.f0method for this run. Default: config value or rmvpe.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override audio and F0 worker counts. RMVPE defaults to 1 worker.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Override runtime.device for F0 and feature stages. Default: config runtime.device",
    )
    parser.add_argument(
        "--is-half",
        dest="is_half",
        action="store_true",
        help="Force half precision where supported for F0 and feature stages.",
    )
    parser.set_defaults(is_half=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be >= 1")

    try:
        stages = parse_stage_list(args.stages)
    except ValueError as exc:
        parser.error(str(exc))

    project = load_project_config(
        args.config,
        overrides=parse_hparams_overrides(args.hparams),
        reset=args.reset,
    )
    f0method = args.f0method or project.get("preprocess", {}).get("f0method") or "rmvpe"
    if f0method not in F0_METHODS:
        parser.error(f"--f0method must be one of: {', '.join(F0_METHODS)}")

    device_override = None if args.device == "auto" else args.device
    run_pipeline(
        project,
        stages,
        f0method,
        args.workers,
        device_override=device_override,
        is_half_override=args.is_half,
    )


if __name__ == "__main__":
    main()
