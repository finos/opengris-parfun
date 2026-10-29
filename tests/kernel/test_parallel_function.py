import unittest
from typing import List
from unittest.mock import patch

import parfun as pf
from parfun.backend.local_single_process import LocalSingleProcessSession
from parfun.kernel.parallel_function import ParallelFunction, is_nested_parallelism


@pf.parallel(split=pf.per_argument(values=pf.py_list.by_chunk), combine_with=sum, fixed_partition_size=1)
def custom_parallel_sum(values: List[int]) -> int:
    return sum(values)


@pf.parallel(split=pf.per_argument(values=pf.py_list.by_chunk), combine_with=all, fixed_partition_size=1)
def check_nested_flag_in_worker(values: List[int]) -> bool:
    return is_nested_parallelism()


class TestParallelFunction(unittest.TestCase):
    def setUp(self) -> None:
        pf.set_parallel_backend("local_multiprocessing")

    def test_validate_signature(self):
        # These are valid:

        ParallelFunction(
            function=lambda x, y: x + y,  # type: ignore[misc, arg-type]
            function_name="lambda",
            split=pf.per_argument(x=lambda x: [(x,)]),  # type: ignore[arg-type, return-value]
            combine_with=sum,
        )

        ParallelFunction(
            function=lambda *args, **kwargs: tuple(),  # type: ignore[misc, arg-type]
            function_name="lambda",
            split=pf.all_arguments(lambda x, y: [(x, y)]),  # type: ignore[arg-type, return-value]
            combine_with=sum,
        )

    def test_submitted_worker_function_preserves_original_name(self):
        submitted_names: List[str] = []
        original_submit = LocalSingleProcessSession.submit

        def recording_submit(session_self, fn, *args, **kwargs):
            submitted_names.append(getattr(fn, "__name__", ""))
            return original_submit(session_self, fn, *args, **kwargs)

        with pf.set_parallel_backend_context("local_single_process"):
            with patch.object(LocalSingleProcessSession, "submit", recording_submit):
                result = custom_parallel_sum([1, 2, 3, 4])

        self.assertEqual(result, 10)
        self.assertEqual(submitted_names, ["custom_parallel_sum"] * 4)

    def test_worker_function_preserves_nested_parallelism_and_multiprocessing(self):
        with pf.set_parallel_backend_context("local_multiprocessing"):
            self.assertEqual(custom_parallel_sum([1, 2, 3, 4]), 10)
            self.assertTrue(check_nested_flag_in_worker([1, 2, 3]))

    def test_scaler_backend_submits_original_function_name(self):
        from parfun.backend.scaler import ScalerSession

        submitted_names: List[str] = []
        original_submit = ScalerSession.submit

        def recording_submit(session_self, fn, *args, **kwargs):
            submitted_names.append(getattr(fn, "__name__", ""))
            return original_submit(session_self, fn, *args, **kwargs)

        with pf.set_parallel_backend_context("scaler_local", n_workers=2):
            with patch.object(ScalerSession, "submit", recording_submit):
                result = custom_parallel_sum([1, 2, 3, 4])

        self.assertEqual(result, 10)
        self.assertEqual(submitted_names, ["custom_parallel_sum"] * 4)


if __name__ == "__main__":
    unittest.main()
