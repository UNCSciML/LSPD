#!/usr/bin/env python3
"""Convert math Parquet datasets to the VERL schema with TTRL prompts.

Accepts either ``prompt`` chat messages and ``reward_model.ground_truth``,
or raw ``problem`` and ``answer`` columns. Existing metadata is preserved.
"""

import argparse
from copy import deepcopy
from pathlib import Path


TTRL_PROMPT_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
DAPO_PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def format_problem(problem: str) -> str:
    """Remove known wrappers and append the exact TTRL suffix once."""
    if not isinstance(problem, str):
        raise ValueError("Problem text must be a string")
    # Preserve whitespace inside the original problem: the training conversion
    # removes the DAPO wrapper literally, including only its two boundary newlines.
    problem = problem.rstrip()
    while True:
        previous = problem
        if problem.endswith(TTRL_PROMPT_SUFFIX):
            problem = problem[: -len(TTRL_PROMPT_SUFFIX)]
        elif problem.endswith(TTRL_PROMPT_SUFFIX.strip()):
            problem = problem[: -len(TTRL_PROMPT_SUFFIX.strip())]
        if problem.startswith(DAPO_PROMPT_PREFIX) and problem.endswith(DAPO_PROMPT_SUFFIX):
            problem = problem[len(DAPO_PROMPT_PREFIX) : -len(DAPO_PROMPT_SUFFIX)]
        if problem == previous:
            break
    if not problem.strip():
        raise ValueError("Problem text must not be empty")
    return problem + TTRL_PROMPT_SUFFIX


def normalize_row(row: dict, index: int, split: str) -> dict:
    """Return a validated, independent row without altering its source metadata."""
    result = deepcopy(row)
    prompt = result.get("prompt")
    if prompt is None:
        if "problem" not in result:
            raise ValueError("Expected a prompt message list or a problem column")
        prompt = [{"role": "user", "content": result["problem"]}]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a nonempty list of chat messages")
    for message in prompt:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Every prompt message must have a valid chat role")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Every prompt message must have nonempty string content")
    if prompt[-1]["role"] != "user":
        raise ValueError("The final prompt message must have the user role")
    prompt[-1]["content"] = format_problem(prompt[-1]["content"])
    result["prompt"] = prompt

    reward_model = result.get("reward_model")
    if reward_model is None:
        reward_model = {"ground_truth": result.get("answer"), "style": "rule"}
    if not isinstance(reward_model, dict):
        raise ValueError("reward_model must be a mapping")
    answer = reward_model.get("ground_truth")
    if answer is None or isinstance(answer, (list, dict, bool)) or not str(answer).strip():
        raise ValueError("Expected a nonempty reward_model.ground_truth or answer")
    reward_model["ground_truth"] = str(answer)
    reward_model.setdefault("style", "rule")
    result["reward_model"] = reward_model

    result["data_source"] = result.get("data_source") or result.get("source") or "math"
    result["ability"] = result.get("ability") or "math"
    extra_info = result.get("extra_info")
    if extra_info is None:
        extra_info = {}
    if not isinstance(extra_info, dict):
        raise ValueError("extra_info must be a mapping")
    extra_info.setdefault("index", result.get("index", index))
    extra_info.setdefault("split", split)
    result["extra_info"] = extra_info
    return result


def prepare_table(input_path: Path, split: str):
    """Read and validate all rows before creating an output file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.read_table(input_path)
    if source.num_rows == 0:
        raise ValueError(f"{split}: input dataset is empty")
    rows = []
    for index, row in enumerate(source.to_pylist()):
        try:
            rows.append(normalize_row(row, index, split))
        except ValueError as error:
            raise ValueError(f"{split} row {index}: {error}") from error
    return pa.Table.from_pylist(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True, type=Path, help="Source training Parquet file")
    parser.add_argument("--val-input", required=True, type=Path, help="Source validation Parquet file")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for train.parquet and val.parquet")
    args = parser.parse_args(argv)
    outputs = [args.output_dir / "train.parquet", args.output_dir / "val.parquet"]
    inputs = [args.train_input, args.val_input]
    if {path.resolve() for path in inputs} & {path.resolve() for path in outputs}:
        parser.error("Output files must differ from the source files")
    if any(path.exists() for path in outputs):
        parser.error("Output files already exist; choose a new output directory or remove them first")

    import pyarrow.parquet as pq

    tables = [prepare_table(path, split) for path, split in zip(inputs, ("train", "val"))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for table, output in zip(tables, outputs):
        pq.write_table(table, output)
        print(f"Wrote {table.num_rows} examples to {output}")


if __name__ == "__main__":
    main()
