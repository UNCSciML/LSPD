<div align="center">

<h1>An RL View of OPD:<br>
Least Square Policy Distillation<br>
for Sample-Efficient LLM Reasoning</h1>

<p>
  <strong>Shangzhe&nbsp;Li</strong><sup>1</sup> &nbsp;&nbsp;
  <strong>Yuxiao&nbsp;Yang</strong><sup>1</sup> &nbsp;&nbsp;
  <strong>Tianrun&nbsp;Yu</strong><sup>2</sup> &nbsp;&nbsp;
  <strong>Kaixiang&nbsp;Zhao</strong><sup>2</sup><br>
  <strong>Xiaoyun&nbsp;Wang</strong><sup>3</sup> &nbsp;&nbsp;
  <strong>Taylor&nbsp;W.&nbsp;Killian</strong><sup>2</sup> &nbsp;&nbsp;
  <strong>Weitong&nbsp;Zhang</strong><sup>1</sup>
</p>

<p>
  <sup>1</sup> University of North Carolina at Chapel Hill<br>
  <sup>2</sup> Brigham Young University &nbsp;&nbsp;
  <sup>3</sup> NVIDIA
</p>

<p>
  <a href="https://arxiv.org/abs/2609.35505"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B" alt="Paper on arXiv"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB" alt="Python 3.12">
  <img src="https://img.shields.io/badge/CUDA-12.8-76B900" alt="CUDA 12.8">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2563EB" alt="License: Apache 2.0"></a>
</p>

<p>
  <a href="https://arxiv.org/abs/2609.35505"><strong>Paper</strong></a> &nbsp;·&nbsp;
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#environment-setup">Setup</a> &nbsp;·&nbsp;
  <a href="#prepare-training-and-evaluation-data">Data</a> &nbsp;·&nbsp;
  <a href="#training">Training</a> &nbsp;·&nbsp;
  <a href="#evaluation">Evaluation</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

</div>

---

## Overview

Research implementation of **LSPD** and its replay-buffer variant, **LSPD-RB**, for sample-efficient language-model reasoning.

LSPD matches student and teacher token log probabilities with a robust squared penalty and encourages student entropy. LSPD-RB reuses previously collected trajectories and fixed teacher targets through a FIFO replay buffer. Both recipes include the TTRL math prompt and teacher/student EOS correction.

This release provides environment setup, public data preparation, training launchers, automatic **AMC23, AIME24, and AIME25** evaluation, local score reporting, and CPU tests. The default teacher is `Qwen/Qwen3-4B`; the student is `Qwen/Qwen3-1.7B-Base` in non-thinking mode.

## Environment setup

### Requirements

| Component | Reference configuration |
| :--- | :--- |
| Platform | Linux x86_64 |
| Runtime | Python 3.12 · CUDA 12.8 |
| Hardware | 4 × NVIDIA RTX PRO 6000, 96 GB per GPU |
| GPU allocation | 2 student/rollout GPUs + 2 teacher GPUs |

A compatible NVIDIA driver, a CUDA toolkit with `nvcc`, and a C++ compiler are required to build FlashAttention. Smaller GPUs require memory/offload or sequence-length adjustments.

### Installation

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/setup_env.sh --gpu
```

The setup installs PyTorch 2.8.0, vLLM 0.11.0, Transformers 4.57.1, FlashAttention 2.8.1, and the pinned requirements. See [requirements.txt](requirements.txt) for the runtime dependencies. The launchers set `PYTHONPATH` to the included `verl/verl` source, which contains the LSPD changes.

<details>
<summary><strong>Dependency and model-loading notes</strong></summary>

The CUDA/PyTorch pairing follows the [vLLM 0.11.0 installation instructions](https://github.com/vllm-project/vllm/blob/v0.11.0/docs/getting_started/installation/gpu/cuda.inc.md) and [versioned dependency pins](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt).

Models are downloaded from Hugging Face on first use. You can supply local directories using `--student` and `--teacher`. Before training, the launcher checks vocabulary compatibility, the non-thinking chat template, and EOS IDs.

</details>

## Prepare training and evaluation data

Download and prepare the reference data:

```bash
python scripts/prepare_benchmarks.py --output-dir data
```

The script creates `data/train.parquet`, `data/val.parquet`, and `data/manifest.json`. The manifest records source URLs, row counts, and SHA-256 checksums.

| Split | Dataset | Rows |
| :--- | :--- | ---: |
| Training | DAPO-Math-17k | 17,917 |
| Validation | AMC23 | 83 |
| Validation | AIME24 | 30 |
| Validation | AIME25 | 30 |

Validation datasets retain separate `data_source` labels for per-benchmark scores. Downloads use a [pinned public OPD source revision](https://github.com/THUNLP/OPD/tree/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/datasets) with integrity checks.

> **AMC23 data variant:** This release uses the 83-row copy from the source experiments, which differs from some public 40-row AMC23 releases.

The preparation script applies the same TTRL suffix to training and evaluation:

```text
 Please reason step by step, and put your final answer within \boxed{}.
```

Use a fresh output directory for each preparation; existing outputs are not overwritten. Downloaded datasets and weights are excluded from version control.

<details>
<summary><strong>Local datasets and custom data</strong></summary>

For local copies of the reference datasets, use:

```bash
python scripts/prepare_benchmarks.py \
  --train-input raw/train.parquet \
  --amc23-input raw/AMC23.parquet \
  --aime24-input raw/AIME24.parquet \
  --aime25-input raw/AIME25.parquet \
  --output-dir data
```

For a custom training/validation pair, `scripts/prepare_data.py` also accepts `problem`/`answer` columns or VERL `prompt`/`reward_model.ground_truth` rows. Benchmark labels must remain distinct for separate scores.

</details>

## Training

Run each command on a node with all four allocated GPUs visible. Launch the methods separately, or use separate GPU allocations.

### LSPD

Train with automatic evaluation on all three benchmarks:

```bash
bash train_lspd.sh \
  --train-data data/train.parquet --val-data data/val.parquet \
  --steps 100 --test-freq 5 --save-freq 5 \
  --output-dir outputs/lspd
```

### LSPD-RB

Train with replay-buffer reuse and the same benchmark evaluations:

```bash
bash train_lspd_rb.sh \
  --train-data data/train.parquet --val-data data/val.parquet \
  --steps 100 --test-freq 5 --save-freq 5 \
  --output-dir outputs/lspd-rb
```

Validation runs **before training and every five rollout steps**. Each benchmark prompt receives **16 sampled responses**, with temperature **0.7**, top-p **0.95**, and a maximum of **7,168 response tokens**. The TTRL grader checks boxed answers. Evaluation uses the current student; the training objective uses teacher probabilities and student entropy.

> **Rollout budget:** The paper reports LSPD-RB results after **10 rollout steps**. Use `--steps 10` for that budget; the 100-step default exposes the full training-curve recipe. One rollout step collects **64 prompts × 4 responses**, not one optimizer update.

### Default configuration

| Setting | LSPD | LSPD-RB |
| :--- | :--- | :--- |
| Student / teacher | Qwen3-1.7B-Base / Qwen3-4B | Same |
| Entropy coefficient / Huber threshold | 0.1 / 5 | Same |
| Fresh prompts × responses | 64 × 4 | Same |
| Trajectories per optimizer update | 64 (16 prompts × 4 responses) | 64 from replay |
| Optimizer updates per rollout step | 4 | 256 |
| Replay capacity | — | 65,536 trajectories |
| Learning rate / gradient norm limit | 1e-6 / 1.0 | Same |
| Prompt / response token limit | 1,024 / 7,168 | Same |
| Actor parameter / optimizer CPU offload | Enabled | Disabled |
| Student / teacher GPUs | 2 / 2 | Same |
| Checkpoint / evaluation interval | Every 5 rollout steps | Same |

The replay buffer stores complete trajectories with detached teacher log probabilities on CPU, samples uniformly without replacement within each update, and starts updating after the first rollout. Student log probabilities and entropy are recomputed. **Replay contents are not checkpointed.**

New launches start from the supplied student model with automatic checkpoint resumption disabled. Use a fresh output directory per run. Paths are relative to the working directory. The active environment's `python` is used; set `PYTHON=/path/to/python` to select another interpreter. Ray starts a local cluster for the launch.

<details>
<summary><strong>Recipe adjustments and advanced options</strong></summary>

```bash
# Unbounded squared loss instead of the default Huber penalty.
bash train_lspd.sh --penalty squared --output-dir outputs/lspd-squared

# Adjust replay reuse.
bash train_lspd_rb.sh --replay-capacity 65536 \
  --replay-batch-size 64 --replay-updates 256 --output-dir outputs/lspd-rb-custom

# Inspect all launcher options.
python scripts/train.py --help
```

Additional Hydra `key=value` arguments can tune backend settings. The loader makes one pass by default; 100 full steps require at least 6,400 usable training prompts after prompt-length filtering. Increase `trainer.total_epochs` for smaller training sets. Method-defining settings are checked to prevent accidental changes to the objective.

</details>

## Evaluation

Both launchers evaluate AMC23, AIME24, and AIME25 automatically when using the prepared `data/val.parquet`. Scores are printed to the console and saved in each run's `metrics.jsonl`.

```bash
python scripts/summarize_eval.py outputs/lspd/metrics.jsonl
python scripts/summarize_eval.py outputs/lspd-rb/metrics.jsonl

# Include the baseline (step 0) and all later evaluations.
python scripts/summarize_eval.py outputs/lspd-rb/metrics.jsonl --all
```

| Metric | Definition |
| :--- | :--- |
| **Avg@16** | Mean correctness across 16 responses per problem, averaged over problems. |
| **Pass@16** | Fraction of problems with at least one correct response among the 16 samples. |

Raw metrics are fractions in `[0, 1]`; the summary reports **percentages**.

For example, AMC23 uses `val-core/AMC23/acc/mean@16` for Avg@16 and `val-core/AMC23/acc/pass@16` for Pass@16. The other benchmark names replace `AMC23` in these keys. Use `--eval-samples N` to change the number of responses; keep 16 for comparison to Table 1.

The backend also logs bootstrap `best@k` diagnostics. These are different statistics from empirical Pass@16; the summary uses the explicitly calculated pass metric.

## Method and EOS alignment

### Distillation objective

For each sampled response token, the implementation uses:

```text
d = log p_student(token | prefix) - log p_teacher(token | prefix)
loss = mean_responses(mean_valid_tokens(rho(d) - alpha * H(student)))

rho(d) = d^2                         if |d| <= delta
         2 * delta * |d| - delta^2   otherwise
```

The defaults are `alpha=0.1` and `delta=5`, with no factor of one half. Teacher targets are detached. Training samples the full student categorical distribution (`temperature=1`, `top_p=1`, `top_k=-1`) without an extra probability weight or importance-sampling factor.

### EOS alignment

Generation stops on student EOS `151643` or teacher EOS `151645`. When scoring with the teacher, teacher EOS probability mass is merged into the student EOS entry before gathering token log probabilities. The response mask recognizes both stopping tokens. Internal `opd_*` configuration names are retained for backend compatibility.

## Development and checks

### CPU checks

For CPU development without the training runtime, create and activate a fresh Python 3.12 environment, then run `bash scripts/setup_env.sh --cpu`. The GitHub Actions workflow uses this setup. CPU checks do not validate distributed GPU execution.

Print the complete launch commands without dependencies, downloads, or GPUs:

```bash
bash train_lspd.sh --dry-run
bash train_lspd_rb.sh --dry-run
```

Compose both configurations and run the CPU tests in the prepared environment:

```bash
python -m pip install -r requirements-dev.txt
python scripts/train.py --check-config
python scripts/train.py --method LSPD-RB --check-config
PYTHONPATH="$PWD/verl:$PWD" PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -p no:cacheprovider tests
```

### GPU smoke test

For a brief GPU training check on the same four-GPU layout:

```bash
bash train_lspd_rb.sh \
  --steps 1 --train-batch-size 4 --responses 2 \
  --replay-batch-size 4 --replay-updates 1 \
  --max-prompt-length 512 --max-response-length 512 \
  --output-dir outputs/smoke \
  trainer.val_before_train=false trainer.test_freq=-1
```

This smoke command explicitly disables evaluation to keep it short; the normal training commands above enable all three benchmarks. Use data with prompts that fit the reduced token limit.

<details>
<summary><strong>Release verification and scope</strong></summary>

Release verification: a fresh Python 3.12 CPU setup passed dependency checks and all **60 tests**. Both training configurations, public dataset downloads/checksums, and data conversion were checked. GPU dependency resolution was checked, but a fresh GPU installation, FlashAttention build, and distributed training run were not performed while packaging this release.

</details>

## Repository layout

```text
paper/                        Included research manuscript
lspd/                         Model-pair preflight checks
scripts/setup_env.sh          Environment installation
scripts/prepare_benchmarks.py Reference data download and conversion
scripts/prepare_data.py       Custom Parquet conversion
scripts/train.py              Shared LSPD / LSPD-RB launcher
scripts/summarize_eval.py     Per-benchmark score summaries
train_lspd.sh                 LSPD entry point
train_lspd_rb.sh              LSPD-RB entry point
tests/                        CPU regression tests
verl/                         Modified VERL training runtime
.github/workflows/            CPU continuous integration
```

## Citation

Please cite the accompanying manuscript when using this implementation. Machine-readable metadata is provided in [CITATION.cff](CITATION.cff).

```bibtex
@article{li2026rlview,
  title={An RL View of OPD: Least Square Policy Distillation for Sample-Efficient LLM Reasoning},
  author={Shangzhe Li and Yuxiao Yang and Tianrun Yu and Kaixiang Zhao and Xiaoyun Wang and Taylor W. Killian and Weitong Zhang},
  year={2026},
  journal={arXiv preprint 2609.35505},
}
```

## License and acknowledgments

Source code is released under [Apache 2.0](LICENSE). This implementation builds on the [OPD codebase](https://github.com/THUNLP/OPD), [VERL](https://github.com/verl-project/verl), and the TTRL/VERL math-grading recipe. Third-party notices are preserved in the source and summarized in [NOTICE](NOTICE). The software license does not assign a new license to the included manuscript; downloaded models and datasets retain their respective terms.

See [CONTRIBUTING.md](CONTRIBUTING.md) for bug reports, development checks, and contributions.
