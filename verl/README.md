# LSPD training backend

This directory contains the modified VERL 0.7.0.dev runtime used by LSPD and
LSPD-RB. The launchers import this source directly so training uses the included
loss, replay, teacher EOS correction, and evaluation implementations. See the
[top-level README](../README.md) for setup and training commands.

The common distributed training, model, worker, and configuration modules are
retained together to preserve their imports. They contain upstream framework
utilities beyond the two exposed training methods. No upstream experiment
recipes, datasets, model weights, run outputs, or repository history are included.

VERL is third-party software under the Apache License 2.0; see [LICENSE](LICENSE)
and the release's [NOTICE](../NOTICE). Copyright and embedded license notices
in its source files are preserved.
