#!/usr/bin/env bash
# Local venv for driving `flyte run` and for smoke_test.py. The pods build their own
# image from config.py; this is only the host side.
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.12 .venv
# torch first, on its own, from the cu130 index. Anything else resolving torch from
# plain PyPI first gets the CPU-only aarch64 wheel and cuda silently disappears.
uv pip install --python .venv/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt
echo "ok: $(.venv/bin/python -c 'import torch;print(torch.__version__, torch.cuda.is_available())')"
