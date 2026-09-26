"""Prompt and schema checks that do not require GPU libraries."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
_SPEC = importlib.util.spec_from_file_location("prepare_data", _SCRIPT)
prepare_data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_data)


class PrepareDataTests(unittest.TestCase):
    def test_dapo_and_repeated_ttrl_wrappers_are_removed(self):
        problem = "What is 1 + 1?"
        wrapped = prepare_data.DAPO_PROMPT_PREFIX + problem + prepare_data.DAPO_PROMPT_SUFFIX
        expected = problem + prepare_data.TTRL_PROMPT_SUFFIX
        self.assertEqual(prepare_data.format_problem(wrapped), expected)
        self.assertEqual(prepare_data.format_problem(expected), expected)
        self.assertEqual(
            prepare_data.format_problem(wrapped + prepare_data.TTRL_PROMPT_SUFFIX * 2), expected
        )

    def test_existing_metadata_and_system_message_are_preserved(self):
        row = {
            "prompt": [
                {"role": "system", "content": "Solve math problems."},
                {"role": "user", "content": "What is 1 + 1?"},
            ],
            "reward_model": {"ground_truth": "2", "style": "rule"},
            "data_source": "sample_math",
            "source": "sample_source",
            "ability": "MATH",
            "extra_info": {"index": "example-7", "split": "test", "difficulty": 1},
        }
        normalized = prepare_data.normalize_row(row, 0, "val")
        for key in ("reward_model", "data_source", "source", "ability", "extra_info"):
            self.assertEqual(normalized[key], row[key])
        self.assertEqual(normalized["prompt"][0], row["prompt"][0])
        self.assertEqual(row["prompt"][-1]["content"], "What is 1 + 1?")

    def test_dapo_problem_boundary_whitespace_is_preserved(self):
        problem = "\nCompute 1 + 1.\n"
        wrapped = prepare_data.DAPO_PROMPT_PREFIX + problem + prepare_data.DAPO_PROMPT_SUFFIX
        expected = problem + prepare_data.TTRL_PROMPT_SUFFIX
        self.assertEqual(prepare_data.format_problem(wrapped), expected)
        self.assertEqual(prepare_data.format_problem(expected), expected)

    def test_raw_schema_and_invalid_rows(self):
        normalized = prepare_data.normalize_row({"problem": "Compute 2 - 2.", "answer": 0}, 3, "train")
        self.assertEqual(normalized["reward_model"]["ground_truth"], "0")
        self.assertEqual(normalized["extra_info"], {"index": 3, "split": "train"})
        with self.assertRaisesRegex(ValueError, "ground_truth"):
            prepare_data.normalize_row({"problem": "Compute 2 - 2."}, 0, "train")
        with self.assertRaisesRegex(ValueError, "empty"):
            prepare_data.format_problem(prepare_data.TTRL_PROMPT_SUFFIX)
        with self.assertRaisesRegex(ValueError, "user role"):
            prepare_data.normalize_row(
                {"prompt": [{"role": "assistant", "content": "2"}], "answer": "2"}, 0, "train"
            )

    def test_parquet_round_trip_and_source_protection(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            pq.write_table(pa.Table.from_pylist([{"problem": "Compute 1 + 1.", "answer": "2"}]), source)
            output_dir = root / "formatted"
            args = ["--train-input", str(source), "--val-input", str(source), "--output-dir", str(output_dir)]
            prepare_data.main(args)
            for split in ("train", "val"):
                row = pq.read_table(output_dir / f"{split}.parquet").to_pylist()[0]
                self.assertEqual(row["prompt"][0]["content"], "Compute 1 + 1." + prepare_data.TTRL_PROMPT_SUFFIX)
                self.assertEqual(row["reward_model"]["ground_truth"], "2")
                self.assertEqual(row["extra_info"]["split"], split)
            self.assertEqual(pq.read_table(source).column_names, ["problem", "answer"])
            with self.assertRaises(SystemExit):
                prepare_data.main(args)


if __name__ == "__main__":
    unittest.main()
