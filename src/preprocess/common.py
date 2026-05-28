from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any


def append_log(
    message: str,
    log_path: str | Path | None = None,
    *,
    handle=None,
    echo: bool = True,
) -> None:
    if echo:
        print(message)
    if handle is not None:
        handle.write(f"{message}\n")
        handle.flush()
        return
    if log_path is None:
        return
    with open(log_path, "a+", encoding="utf-8") as file_obj:
        file_obj.write(f"{message}\n")
        file_obj.flush()


def log_message(
    log_path: str | Path | None,
    message: str,
    *,
    handle=None,
    echo: bool = True,
) -> None:
    append_log(message, log_path, handle=handle, echo=echo)


def shard_items(items: list[Any], shard_count: int):
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    return [items[index::shard_count] for index in range(shard_count)]


def run_sharded_processes(
    shards,
    target,
    build_args,
    *,
    error_label: str,
) -> None:
    processes: list[multiprocessing.Process] = []
    for shard in shards:
        process = multiprocessing.Process(target=target, args=build_args(shard))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{error_label} {process.pid} exited with {process.exitcode}")


def run_worker_shards(
    items: list[Any],
    worker_count: int,
    target,
    build_args,
    *,
    error_label: str,
    parallel: bool = True,
) -> None:
    shards = shard_items(items, worker_count)
    if not parallel:
        for shard in shards:
            target(*build_args(shard))
        return
    if worker_count == 1:
        target(*build_args(shards[0]))
        return
    run_sharded_processes(
        shards,
        target,
        build_args,
        error_label=error_label,
    )
