from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
from dataclasses import dataclass
from pathlib import Path

from configs.project_config import load_project_config, parse_hparams_overrides
from src.preprocess import f0 as f0_stage


DEFAULT_STAGES = ("audio", "f0", "features", "filelist")
VALID_STAGES = set(DEFAULT_STAGES)
MANIFEST_NAME = "preprocess_manifest.jsonl"
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
F0_METHODS = {"pm", "harvest", "dio", "rmvpe"}
FULL_FILELIST_NAME = "filelist.txt"
TRAIN_FILELIST_NAME = "train_filelist.txt"
VAL_FILELIST_NAME = "val_filelist.txt"


@dataclass(frozen=True)
class DatasetItem:
    source_path: Path
    speaker_id: int
    index: int


def _log_message(message: str, log_path: str | Path | None = None) -> None:
    print(message)
    if log_path is not None:
        with open(log_path, "a+", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
            handle.flush()


def parse_stage_list(raw_stages: str | None) -> tuple[str, ...]:
    if raw_stages in (None, ""):
        return DEFAULT_STAGES

    stages = tuple(
        stage.strip().lower() for stage in raw_stages.split(",") if stage.strip()
    )
    if not stages:
        raise ValueError("--stages cannot be empty")

    invalid = sorted(set(stages) - VALID_STAGES)
    if invalid:
        raise ValueError(
            f"Unsupported stage(s): {', '.join(invalid)}. "
            f"Expected any of: {', '.join(DEFAULT_STAGES)}"
        )
    return tuple(dict.fromkeys(stages))


def _is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def _direct_audio_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if _is_audio_file(path))


def _visible_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _speaker_embed_dim(project: dict) -> int:
    value = project.get("model", {}).get("spk_embed_dim", 1)
    try:
        spk_embed_dim = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model.spk_embed_dim must be an integer, got: {value!r}") from exc
    if spk_embed_dim < 1:
        raise ValueError(f"model.spk_embed_dim must be >= 1, got: {spk_embed_dim}")
    return spk_embed_dim


def discover_dataset_items(
    dataset_dir: str | Path,
    spk_embed_dim: int,
) -> list[DatasetItem]:
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_dir does not exist or is not a directory: {root}")

    top_level_files = _direct_audio_files(root)
    speaker_dirs = _visible_dirs(root)
    if top_level_files and speaker_dirs:
        raise ValueError(
            "dataset_dir cannot mix top-level audio files and speaker subdirectories"
        )

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
            f"Max speaker directory id {max_speaker_id} exceeds "
            f"model.spk_embed_dim={spk_embed_dim}"
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


def _run_audio_items_worker(
    item_payload: list[tuple[str, int]],
    sampling_rate: int,
    preprocess_dir: str,
) -> None:
    from src.preprocess.audio import AudioPreprocessor

    worker = AudioPreprocessor(
        sampling_rate,
        preprocess_dir,
        noparallel=True,
    )
    failures = 0
    for source_path, item_index in item_payload:
        if not worker.pipeline(source_path, item_index):
            failures += 1
    if failures:
        raise RuntimeError(f"{failures} audio preprocessing item(s) failed")


def _output_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem.rsplit("_", 1)[1]), path.name
    except (IndexError, ValueError):
        return 0, path.name


def _manifest_records(
    items: list[DatasetItem],
    preprocess_dir: str | Path,
) -> list[dict]:
    preprocess_path = Path(preprocess_dir)
    gt_wavs_dir = preprocess_path / "0_gt_wavs"
    wavs16k_dir = preprocess_path / "1_16k_wavs"

    records: list[dict] = []
    for item in items:
        for gt_wav in sorted(
            gt_wavs_dir.glob(f"{item.index}_*.wav"),
            key=_output_sort_key,
        ):
            wav16k = wavs16k_dir / gt_wav.name
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


def write_preprocess_manifest(
    items: list[DatasetItem],
    preprocess_dir: str | Path,
) -> Path:
    records = _manifest_records(items, preprocess_dir)
    if not records:
        raise RuntimeError("Audio preprocessing produced no manifest rows")

    manifest_path = Path(preprocess_dir) / MANIFEST_NAME
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    manifest_path.write_text(f"{payload}\n", encoding="utf-8")
    _log_message(f"wrote manifest rows={len(records)}: {manifest_path}")
    return manifest_path


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


def _sample_rate(project: dict) -> str:
    return str(
        project.get("sample_rate", project.get("selectors", {}).get("sample_rate", ""))
    ).lower()


def _legacy_gt_wav_for(wav16k: Path, gt_wavs_dir: Path, project: dict) -> Path | None:
    same_name = gt_wavs_dir / wav16k.name
    if same_name.is_file():
        return same_name

    same_stem = gt_wavs_dir / f"{wav16k.stem}.wav"
    if same_stem.is_file():
        return same_stem

    sample_rate = _sample_rate(project)
    if sample_rate:
        sample_rate_match = gt_wavs_dir / f"{wav16k.stem}{sample_rate}.wav"
        if sample_rate_match.is_file():
            return sample_rate_match

    gt_wavs = sorted(gt_wavs_dir.glob("*.wav"))
    prefix_matches = [
        path for path in gt_wavs if path.stem.lower().startswith(wav16k.stem.lower())
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(gt_wavs) == 1:
        return gt_wavs[0]
    return None


def _legacy_manifest_records(project: dict, preprocess_dir: Path) -> list[dict]:
    gt_wavs_dir = preprocess_dir / "0_gt_wavs"
    wavs16k_dir = preprocess_dir / "1_16k_wavs"
    records: list[dict] = []

    for wav16k in sorted(wavs16k_dir.glob("*.wav")):
        gt_wav = _legacy_gt_wav_for(wav16k, gt_wavs_dir, project)
        if gt_wav is None:
            continue
        records.append(
            {
                "source_path": str(wav16k.resolve()),
                "speaker_id": 0,
                "gt_wav": str(gt_wav.resolve()),
                "wav16k": str(wav16k.resolve()),
            }
        )

    if not records:
        raise FileNotFoundError(
            f"Missing {preprocess_dir / MANIFEST_NAME} and no legacy single-speaker "
            "audio pairs were found"
        )

    print(
        f"warning: {MANIFEST_NAME} is missing; using legacy single-speaker "
        f"filelist scan rows={len(records)}"
    )
    return records


def _if_f0(project: dict) -> int:
    return int(project.get("if_f0", project.get("selectors", {}).get("if_f0", 1)))


def _version(project: dict) -> str:
    return str(project.get("version", project.get("selectors", {}).get("version", "v2")))


def _feature_dir(project: dict, preprocess_dir: Path) -> Path:
    return preprocess_dir / (
        "3_feature256" if _version(project) == "v1" else "3_feature768"
    )


def _build_filelist_rows(
    project: dict,
    records: list[dict],
) -> tuple[list[str], int]:
    preprocess_dir = Path(project["preprocess_dir"])
    feature_dir = _feature_dir(project, preprocess_dir)
    use_f0 = _if_f0(project) == 1
    rows: list[str] = []
    skipped = 0
    for record in records:
        gt_wav = Path(record["gt_wav"])
        wav16k = Path(record["wav16k"])
        feature_path = feature_dir / wav16k.with_suffix(".npy").name
        sid = int(record["speaker_id"])

        required_paths = [gt_wav, feature_path]
        if use_f0:
            coarse_f0 = preprocess_dir / "2a_f0" / f"{wav16k.name}.npy"
            nsf_f0 = preprocess_dir / "2b-f0nsf" / f"{wav16k.name}.npy"
            required_paths.extend([coarse_f0, nsf_f0])

        if not all(path.is_file() for path in required_paths):
            skipped += 1
            continue

        if use_f0:
            rows.append(
                "|".join(map(str, [gt_wav, feature_path, coarse_f0, nsf_f0, sid]))
            )
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
        raise ValueError(
            f"preprocess.validation_seed must be an integer, got: {value!r}"
        ) from exc


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
                "Validation split produced no validation samples. "
                "Add more data or adjust preprocess.validation_split."
            )

    train_lines: list[str] = []
    val_lines = []
    for sid in sorted(speaker_groups.keys(), key=_speaker_sort_key):
        train_lines.extend(grouped_train_lines.get(sid, []))
        val_lines.extend(grouped_val_lines.get(sid, []))

    return train_lines, val_lines


def generate_filelist(project: dict) -> tuple[Path, int, int]:
    preprocess_dir = Path(project["preprocess_dir"])
    try:
        records = load_preprocess_manifest(preprocess_dir)
    except FileNotFoundError:
        records = _legacy_manifest_records(project, preprocess_dir)

    rows, skipped = _build_filelist_rows(project, records)

    if not rows:
        raise RuntimeError(
            "No valid preprocess samples found for filelist. "
            f"manifest_rows={len(records)}, skipped={skipped}"
        )

    filelist_path = preprocess_dir / FULL_FILELIST_NAME
    filelist_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote filelist rows={len(rows)}, skipped={skipped}: {filelist_path}")

    train_lines, val_lines = split_filelist_rows(
        rows,
        _resolve_validation_split(project),
        _resolve_validation_seed(project),
    )
    train_filelist_path = preprocess_dir / TRAIN_FILELIST_NAME
    val_filelist_path = preprocess_dir / VAL_FILELIST_NAME
    train_filelist_path.write_text(
        "\n".join(train_lines) + "\n", encoding="utf-8"
    )
    val_filelist_path.write_text(
        "\n".join(val_lines) + "\n", encoding="utf-8"
    )
    print(f"wrote train filelist rows={len(train_lines)}: {train_filelist_path}")
    print(f"wrote val filelist rows={len(val_lines)}: {val_filelist_path}")
    return filelist_path, len(rows), skipped


def _resolve_audio_workers(project: dict, workers_override: int | None) -> int:
    return int(workers_override if workers_override is not None else project["n_cpu"])


def _resolve_f0_workers(
    project: dict,
    f0method: str,
    workers_override: int | None,
) -> int:
    if workers_override is not None:
        return int(workers_override)
    return 1 if f0method == "rmvpe" else int(project["n_cpu"])


def run_audio_stage(project: dict, workers_override: int | None = None) -> Path:
    preprocess_dir = Path(project["preprocess_dir"])
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    items = discover_dataset_items(project["dataset_dir"], _speaker_embed_dim(project))
    sampling_rate = int(project["data"]["sampling_rate"])
    preprocess_config = project.get("preprocess", {})
    workers = min(_resolve_audio_workers(project, workers_override), len(items))
    workers = max(workers, 1)
    noparallel = bool(preprocess_config.get("noparallel", False))
    log_path = preprocess_dir / "preprocess.log"

    _log_message(
        f"start audio preprocess, items={len(items)}, workers={workers}, "
        f"noparallel={noparallel}, mode=prepared-audio",
        log_path,
    )

    payload = [(str(item.source_path), item.index) for item in items]
    if noparallel or workers == 1:
        _run_audio_items_worker(
            payload,
            sampling_rate,
            str(preprocess_dir),
        )
    else:
        processes: list[multiprocessing.Process] = []
        for worker_index in range(workers):
            process = multiprocessing.Process(
                target=_run_audio_items_worker,
                args=(
                    payload[worker_index::workers],
                    sampling_rate,
                    str(preprocess_dir),
                ),
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"audio preprocess worker {process.pid} exited with {process.exitcode}"
                )

    manifest_path = write_preprocess_manifest(items, preprocess_dir)
    _log_message("end audio preprocess", log_path)
    return manifest_path


def run_f0_stage(
    project: dict,
    f0method: str,
    workers_override: int | None = None,
) -> None:
    if _if_f0(project) == 0:
        print("skip f0 stage because selectors.if_f0=0")
        return

    if f0method not in F0_METHODS:
        raise ValueError(f"Unsupported f0method={f0method!r}")

    exp_dir = Path(project["preprocess_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "extract_f0_feature.log"
    workers = _resolve_f0_workers(project, f0method, workers_override)

    i_gpu = ""
    device_request = str(project["device"])
    if device_request.startswith("cuda:"):
        i_gpu = device_request.split(":", 1)[1]

    runtime_args = argparse.Namespace(
        f0method=f0method,
        i_gpu=i_gpu,
        is_half=bool(project["is_half"]),
    )
    device, is_half, _ = f0_stage.resolve_runtime(runtime_args, log_path)
    pretrain_root = str(
        project.get("pretrain_root", os.getenv("pretrain_root", "pretrain"))
    )
    paths = f0_stage.build_paths(exp_dir)

    f0_stage.log_message(
        log_path,
        f"pipeline-f0,method={f0method},workers={workers},device={device},is_half={is_half}",
    )
    if workers == 1:
        f0_stage.run_worker(paths, f0method, device, is_half, pretrain_root, log_path)
        return

    processes: list[multiprocessing.Process] = []
    for worker_index in range(workers):
        process = multiprocessing.Process(
            target=f0_stage.run_worker,
            args=(
                paths[worker_index::workers],
                f0method,
                device,
                is_half,
                pretrain_root,
                log_path,
            ),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"f0 worker {process.pid} exited with code {process.exitcode}")


def run_feature_stage(project: dict) -> None:
    from src.preprocess import features as feature_stage

    exp_dir = Path(project["preprocess_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    device_request = str(project["device"])
    device = feature_stage.resolve_device(
        device_request if device_request.startswith("cuda") else "cpu"
    )
    feature_stage.extract_features(
        exp_dir=exp_dir,
        version=_version(project),
        n_part=1,
        i_part=0,
        device=device,
        is_half=bool(project["is_half"]),
        model_path=str(project["hubert_path"]),
        log_path=exp_dir / "extract_f0_feature.log",
    )


def run_pipeline(
    project: dict,
    stages: tuple[str, ...],
    f0method: str,
    workers_override: int | None = None,
) -> None:
    for stage in stages:
        if stage == "audio":
            run_audio_stage(project, workers_override)
        elif stage == "f0":
            run_f0_stage(project, f0method, workers_override)
        elif stage == "features":
            run_feature_stage(project)
        elif stage == "filelist":
            generate_filelist(project)
        else:  # pragma: no cover - parse_stage_list rejects this.
            raise ValueError(f"Unsupported stage: {stage}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full RVC preprocessing pipeline: audio, F0, HuBERT features, "
            "and filelist generation."
        )
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
    f0method = (
        args.f0method
        or project.get("preprocess", {}).get("f0method")
        or "rmvpe"
    )
    if f0method not in F0_METHODS:
        parser.error("--f0method must be one of: pm, harvest, dio, rmvpe")

    run_pipeline(project, stages, f0method, args.workers)


if __name__ == "__main__":
    main()
