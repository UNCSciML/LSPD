"""Offline regression tests for benchmark labels, prompts, and safe publication."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import prepare_benchmarks as prepare


class PrepareBenchmarksTests(unittest.TestCase):
    def make_inputs(self, root):
        paths = {}
        for name in prepare.SOURCES:
            paths[name] = root / f"{name}.parquet"
            row = {
                "problem": f"A distinct {name} problem: compute 1 + 1.",
                "answer": "2",
                "data_source": "original_source",
                "extra_info": {"index": 42, "split": "original_split"},
            }
            if name == "AIME25":
                row["question"] = row.pop("problem")
            pq.write_table(pa.Table.from_pylist([row]), paths[name])
        return paths

    def test_local_sources_keep_benchmarks_separate_and_training_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.make_inputs(root)
            output = root / "data"
            with patch.object(prepare, "download_source", side_effect=AssertionError("unexpected network")):
                prepare.main([
                    "--output-dir", str(output),
                    "--train-input", str(inputs["DAPO"]),
                    "--amc23-input", str(inputs["AMC23"]),
                    "--aime24-input", str(inputs["AIME24"]),
                    "--aime25-input", str(inputs["AIME25"]),
                ])
            train = pq.read_table(output / "train.parquet").to_pylist()
            validation = pq.read_table(output / "val.parquet").to_pylist()
            self.assertEqual([row["data_source"] for row in validation], list(prepare.EVAL_SOURCES))
            self.assertEqual([row["data_source"] for row in train], ["math_dapo"])
            self.assertIn("DAPO problem", train[0]["prompt"][0]["content"])
            self.assertEqual(len({row["extra_info"]["index"] for row in validation}), 3)
            for row in train + validation:
                self.assertEqual(row["prompt"][0]["content"].count(prepare.TTRL_PROMPT_SUFFIX), 1)
                self.assertEqual(row["reward_model"]["ground_truth"], "2")
                metadata = json.loads(row["extra_info"]["source_metadata"])
                self.assertEqual(metadata["extra_info"]["index"], 42)
                self.assertEqual(metadata["data_source"], "original_source")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["outputs"]["val.parquet"]["data_sources"], dict.fromkeys(prepare.EVAL_SOURCES, 1))
            for filename in ("train.parquet", "val.parquet"):
                self.assertEqual(manifest["outputs"][filename]["sha256"], prepare.file_sha256(output / filename))

    def test_all_sources_are_validated_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.make_inputs(root)
            pq.write_table(pa.Table.from_pylist([{"problem": "Broken benchmark without an answer"}]), inputs["AIME25"])
            with self.assertRaisesRegex(ValueError, "AIME25 row 0.*ground_truth"):
                prepare.prepare_datasets(root / "data", inputs)
            self.assertFalse((root / "data").exists())

    def test_refuses_existing_output_without_touching_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.make_inputs(root)
            originals = {name: path.read_bytes() for name, path in inputs.items()}
            output = root / "data"
            prepare.prepare_datasets(output, inputs)
            contents = {name: (output / name).read_bytes() for name in prepare.OUTPUT_NAMES}
            with self.assertRaisesRegex(ValueError, "already exist"):
                prepare.prepare_datasets(output, inputs)
            self.assertEqual(contents, {name: (output / name).read_bytes() for name in prepare.OUTPUT_NAMES})
            self.assertEqual(originals, {name: path.read_bytes() for name, path in inputs.items()})

    def test_rejects_empty_and_nonfinite_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.parquet"
            pq.write_table(pa.table({"problem": [], "answer": []}), path)
            with self.assertRaisesRegex(ValueError, "empty"):
                prepare.prepare_rows(path, "AMC23")
            pq.write_table(pa.Table.from_pylist([{"problem": "Compute 1 + 1.", "answer": float("nan")}]), path)
            with self.assertRaisesRegex(ValueError, "finite"):
                prepare.prepare_rows(path, "AMC23")

    def test_download_rejects_unexpected_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(prepare, "urlopen", return_value=io.BytesIO(b"unexpected content")):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    prepare.download_source("AMC23", directory)

    def test_partial_publication_is_rolled_back_without_overwriting_a_racing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.make_inputs(root)
            output = root / "data"
            original_link = prepare.os.link

            def racing_link(source, destination):
                if Path(destination).name == "val.parquet":
                    Path(destination).write_bytes(b"another process owns this file")
                return original_link(source, destination)

            with patch.object(prepare.os, "link", side_effect=racing_link):
                with self.assertRaises(FileExistsError):
                    prepare.prepare_datasets(output, inputs)
            self.assertFalse((output / "train.parquet").exists())
            self.assertFalse((output / "manifest.json").exists())
            self.assertEqual((output / "val.parquet").read_bytes(), b"another process owns this file")


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        unittest.main()
