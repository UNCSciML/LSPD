# Contributing

Bug reports and pull requests are welcome. For a bug report, include the command,
relevant error output, Python/package versions, GPU model and count, and the
smallest example that reproduces the problem. Remove credentials and private
paths from shared logs.

Keep method changes, data preparation changes, and documentation changes focused.
Explain any changes to the objective, EOS handling, replay schedule, or evaluation
protocol so reported results remain comparable.

From the repository root, install the development requirements in your active
environment and run the CPU tests:

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH="$PWD/verl:$PWD" PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -p no:cacheprovider tests
python scripts/train.py --check-config
python scripts/train.py --method LSPD-RB --check-config
```

The test suite covers the loss and gradients, replay behavior, prompt conversion,
EOS correction, math grading, launch configuration, and benchmark reporting.
GPU training checks require the training environment described in the README.
Describe the validation you actually ran in the pull request.

Preserve third-party copyright and license notices. Contributions to the source
code are made under the repository's Apache 2.0 license.
