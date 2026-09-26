#!/usr/bin/env python3
"""Prepare DAPO training data and AMC23/AIME24/AIME25 validation data.

By default, download immutable, checksum-verified files from the public OPD
release. Individual sources can instead be supplied as local Parquet files.
All prompts use the TTRL instruction implemented by ``prepare_data.py``.
No existing output is overwritten. No benchmark examples enter the train split.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

try:
    from .prepare_data import normalize_row, TTRL_PROMPT_SUFFIX
except ImportError:  # Running ``python scripts/prepare_benchmarks.py``.
    from prepare_data import normalize_row, TTRL_PROMPT_SUFFIX


UPSTREAM_REVISION = "ac26e38d6f1572eb027597b48a9f4e01f6915ef8"
UPSTREAM_REPOSITORY = "https://github.com/THUNLP/OPD"
SOURCES = {
    "DAPO": {
        "path": "datasets/dapo-math-17k.parquet",
        "rows": 17917,
        "sha256": "039f3afd689c846985bd2bf58e55a2210a8b08a1a9e1f60dbc07107a5f341925",
    },
    "AMC23": {
        "path": "datasets/test_data/AMC23/test.parquet",
        "rows": 83,
        "sha256": "0f933978194b73fc091bd667af7d9056d3d745c9d9e3577b5adcad77f8f1ae7c",
    },
    "AIME24": {
        "path": "datasets/test_data/AIME24/test.parquet",
        "rows": 30,
        "sha256": "dc3ae15e0236828eb95033a68aab60e19da7ea1afdaad80e26924e5f82d3d96f",
    },
    "AIME25": {
        "path": "datasets/test_data/AIME25/test.parquet",
        "rows": 30,
        "sha256": "14ee9a0e8d79cba76c62247bc5ebee1647e023eccf60e31994e279113777dad2",
    },
}
EVAL_SOURCES = ("AMC23", "AIME24", "AIME25")
OUTPUT_NAMES = ("train.parquet", "val.parquet", "manifest.json")


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(name, directory):
    """Fetch a pinned source without requiring a Hub token or extra client."""
    source = SOURCES[name]
    url = (
        f"https://raw.githubusercontent.com/THUNLP/OPD/{UPSTREAM_REVISION}/"
        f"{source['path']}"
    )
    destination = Path(directory) / f"{name}.parquet"
    request = Request(url, headers={"User-Agent": "LSPD-data-preparation/1.0"})
    with urlopen(request, timeout=120) as response, destination.open("xb") as handle:
        shutil.copyfileobj(response, handle)
    digest = file_sha256(destination)
    if digest != source["sha256"]:
        raise ValueError(f"{name}: downloaded file checksum does not match the pinned release")
    return destination, {
        "kind": "download",
        "url": url,
        "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION,
        "sha256": digest,
    }


def prepare_rows(path, name):
    """Normalize either VERL or raw problem/answer Parquet into one schema.

    Metadata is serialized in ``extra_info.source_metadata`` so distinct source
    schemas cannot lose metadata or produce incompatible Arrow struct types.
    Benchmark names are explicitly assigned for per-benchmark trainer metrics.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if not table.num_rows:
        raise ValueError(f"{name}: input dataset is empty")
    split = "train" if name == "DAPO" else "test"
    rows = []
    for index, row in enumerate(table.to_pylist()):
        try:
            # Also accept the common question/answer representation.
            if row.get("prompt") is None and "problem" not in row and "question" in row:
                row = {**row, "problem": row["question"]}
            reward = row.get("reward_model")
            answer = reward.get("ground_truth") if isinstance(reward, dict) else row.get("answer")
            if isinstance(answer, float) and not math.isfinite(answer):
                raise ValueError("ground_truth must be finite")
            normalized = normalize_row(row, index, split)
            metadata = {
                key: value for key, value in row.items()
                if key not in {"prompt", "problem", "question", "answer", "reward_model", "ability"}
            }
            rows.append({
                "data_source": "math_dapo" if name == "DAPO" else name,
                "prompt": normalized["prompt"],
                "ability": "math",
                "reward_model": {
                    "ground_truth": normalized["reward_model"]["ground_truth"],
                    "style": "rule",
                },
                "extra_info": {
                    "index": f"{name}-{index}",
                    "split": split,
                    "source_metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                },
            })
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} row {index}: {error}") from error
    return rows


def prepare_datasets(output_dir, local_inputs=None):
    """Validate every source, then publish Parquet and provenance atomically.

    Each individual file is published with an exclusive hard link, so even
    concurrent invocations cannot overwrite another process's output.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = Path(output_dir)
    local_inputs = local_inputs or {}
    outputs = [output_dir / name for name in OUTPUT_NAMES]
    if any(os.path.lexists(path) for path in outputs):
        raise ValueError("Output files already exist; choose a new output directory")
    if {Path(path).resolve() for path in local_inputs.values()} & {path.resolve() for path in outputs}:
        raise ValueError("Output files must differ from the source files")
    unknown = set(local_inputs) - set(SOURCES)
    if unknown:
        raise ValueError(f"Unknown source names: {sorted(unknown)}")

    prepared = {}
    provenance = {}
    with tempfile.TemporaryDirectory(prefix="lspd-sources-") as directory:
        for name, source in SOURCES.items():
            if name in local_inputs:
                path = Path(local_inputs[name])
                info = {"kind": "local", "filename": path.name, "sha256": file_sha256(path)}
            else:
                print(f"Downloading {name} from the pinned OPD release...", flush=True)
                path, info = download_source(name, directory)
            rows = prepare_rows(path, name)
            if info["kind"] == "download" and len(rows) != source["rows"]:
                raise ValueError(f"{name}: expected {source['rows']} source rows, got {len(rows)}")
            prepared[name] = rows
            provenance[name] = {**info, "rows": len(rows)}

    train_rows = prepared["DAPO"]
    val_rows = [row for name in EVAL_SOURCES for row in prepared[name]]
    manifest = {
        "schema_version": 1,
        "prompt_suffix": TTRL_PROMPT_SUFFIX,
        "sources": provenance,
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir) as directory:
        staging = Path(directory)
        for filename, rows in (("train.parquet", train_rows), ("val.parquet", val_rows)):
            pq.write_table(pa.Table.from_pylist(rows), staging / filename)
            manifest["outputs"][filename] = {
                "rows": len(rows),
                "data_sources": dict(Counter(row["data_source"] for row in rows)),
                "sha256": file_sha256(staging / filename),
            }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        published = []
        try:
            for target in outputs:
                os.link(staging / target.name, target)
                published.append(target)
        except OSError:
            for target in published:
                if target.exists() and target.samefile(staging / target.name):
                    target.unlink()
            raise
    for filename, info in manifest["outputs"].items():
        print(f"Wrote {info['rows']} examples to {output_dir / filename}: {info['data_sources']}")
    print(f"Saved source provenance to {output_dir / 'manifest.json'}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-input", type=Path, help="Local DAPO Parquet; otherwise download the pinned release")
    for name in EVAL_SOURCES:
        parser.add_argument(
            f"--{name.lower()}-input", type=Path,
            help=f"Local {name} Parquet; otherwise download the pinned release",
        )
    args = parser.parse_args(argv)
    inputs = {
        "DAPO": args.train_input,
        **{name: getattr(args, f"{name.lower()}_input") for name in EVAL_SOURCES},
    }
    try:
        prepare_datasets(args.output_dir, {name: path for name, path in inputs.items() if path is not None})
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
