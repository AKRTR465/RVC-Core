import unittest
from unittest import mock

from src.preprocess.common import run_worker_shards


class PreprocessCommonTest(unittest.TestCase):
    def test_run_worker_shards_serializes_single_worker_without_spawning(self):
        seen = []

        def record(shard):
            seen.append(list(shard))

        run_worker_shards(
            [0, 1, 2],
            1,
            record,
            lambda shard: (shard,),
            error_label="worker",
        )

        self.assertEqual(seen, [[0, 1, 2]])

    def test_run_worker_shards_parallel_false_runs_each_shard_in_process(self):
        seen = []

        def record(shard):
            seen.append(list(shard))

        run_worker_shards(
            [0, 1, 2, 3],
            2,
            record,
            lambda shard: (shard,),
            error_label="worker",
            parallel=False,
        )

        self.assertEqual(seen, [[0, 2], [1, 3]])

    def test_run_worker_shards_delegates_multi_worker_parallel_runs(self):
        with mock.patch("src.preprocess.common.run_sharded_processes") as mocked:
            run_worker_shards(
                [0, 1, 2, 3],
                2,
                lambda shard: None,
                lambda shard: (shard,),
                error_label="worker",
            )

        mocked.assert_called_once()
        shards, target, build_args = mocked.call_args.args
        self.assertEqual(shards, [[0, 2], [1, 3]])
        self.assertEqual(mocked.call_args.kwargs["error_label"], "worker")
        self.assertEqual(build_args(["x"]), (["x"],))
        self.assertIsNotNone(target)


if __name__ == "__main__":
    unittest.main()
