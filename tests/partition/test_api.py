import math
import unittest

import pandas as pd

import parfun as pf
from parfun.kernel.function_signature import NamedArguments
from parfun.partition.utility import with_partition_size


class TestPartitionAPI(unittest.TestCase):
    def test_per_argument(self):
        N = 100
        PARTITION_SIZE = 2
        N_PARTITIONS = math.ceil(N / PARTITION_SIZE)

        def custom_chunk_generator(values):
            for i in range(0, N_PARTITIONS):
                yield values[i * PARTITION_SIZE : (i + 1) * PARTITION_SIZE],

        partitioning_function = pf.per_argument(
            values=pf.py_list.by_chunk,
            df=pf.dataframe.by_row,
            custom=custom_chunk_generator
        )

        xs = [x for x in range(0, N)]
        df = pd.DataFrame({"x^2": [x * x for x in xs]})

        args = NamedArguments(kwargs={"values": xs, "df": df, "custom": xs, "constant": 1})

        non_partitioned_args, partition_generator = partitioning_function(args)

        self.assertEqual(len(non_partitioned_args.keys()), 1)
        self.assertEqual(non_partitioned_args["constant"], 1)

        partitions = list(with_partition_size(partition_generator, partition_size=PARTITION_SIZE))

        self.assertEqual(len(partitions), N_PARTITIONS)

        for i, partition in enumerate(partitions):
            self.assertEqual(len(partition.keys()), 3)

            partition_xs = xs[i * 2 : i * 2 + 2]

            self.assertSequenceEqual(partition["values"], partition_xs)
            self.assertSequenceEqual(list(partition["df"]["x^2"]), [x * x for x in partition_xs])
            self.assertSequenceEqual(partition["custom"], partition_xs)

    def test_multiple_arguments(self):
        partitioning_function = pf.multiple_arguments(("df_1", "df_2"), pf.dataframe.by_group(by="year"))

        df_1 = pd.DataFrame({"year": [2020, 2021, 2020, 2020, 2022], "values": range(0, 5)})

        df_2 = df_1.copy()
        df_2["values"] **= 2

        args = NamedArguments(kwargs={"df_1": df_1, "df_2": df_2, "constant": 2})

        non_partitioned_args, partition_generator = partitioning_function(args)

        self.assertEqual(len(non_partitioned_args.keys()), 1)
        self.assertEqual(non_partitioned_args["constant"], 2)

        partitions = list(with_partition_size(partition_generator, partition_size=1))

        self.assertEqual(len(partitions), df_1["year"].unique().shape[0])

        for partition in partitions:
            self.assertEqual(len(partition.keys()), 2)

            self.assertEqual(partition["df_1"].shape[0], partition["df_2"].shape[0])
            self.assertSequenceEqual(list(partition["df_1"]["values"] ** 2), list(partition["df_2"]["values"]))

    def test_all_arguments(self):
        N = 100
        PARTITION_SIZE = 3
        N_PARTITIONS = math.ceil(N / PARTITION_SIZE)

        partitioning_function = pf.all_arguments(pf.py_list.by_chunk)

        xs = list(range(0, N))
        ys = [x * x for x in xs]

        args = NamedArguments(kwargs={"xs": xs, "ys": ys})

        non_partitioned_args, partition_generator = partitioning_function(args)

        self.assertEqual(len(non_partitioned_args.keys()), 0)

        partitions = list(with_partition_size(partition_generator, partition_size=PARTITION_SIZE))

        self.assertEqual(len(partitions), N_PARTITIONS)

        for partition in partitions:
            self.assertEqual(len(partition.kwargs), 2)

            self.assertLessEqual(len(partition["xs"]), PARTITION_SIZE)

            self.assertSequenceEqual([x * x for x in partition["xs"]], partition["ys"])

    def test_zero_partitions(self):
        # Validates that per_argument, multiple_argument and all_arguments return the original arguments if the
        # arguments can't be partitioned in a least one partition.

        empty_args = NamedArguments(kwargs={"xs": [], "ys": []})  # type: ignore

        # per_argument()

        _, per_arg_partition_generator = pf.per_argument(
            xs=pf.py_list.by_chunk,
            ys=pf.py_list.by_chunk,
        )(empty_args)
        per_arg_partitions = list(with_partition_size(per_arg_partition_generator, partition_size=10))

        self.assertEqual(len(per_arg_partitions), 1)
        self.assertEqual(per_arg_partitions[0], empty_args)

        # multiple_arguments()

        _, multiple_generator = pf.multiple_arguments(
            partition_on=("xs", "ys"),
            partition_with=pf.py_list.by_chunk
        )(empty_args)
        multiple_partitions = list(with_partition_size(multiple_generator, partition_size=10))

        self.assertEqual(len(multiple_partitions), 1)
        self.assertEqual(multiple_partitions[0], empty_args)

        # all_arguments()

        _, all_args_generator = pf.all_arguments(pf.py_list.by_chunk)(empty_args)
        all_args_partitions = list(with_partition_size(all_args_generator, partition_size=10))

        self.assertEqual(len(all_args_partitions), 1)
        self.assertEqual(all_args_partitions[0], empty_args)


if __name__ == "__main__":
    unittest.main()
