#!/usr/bin/env bash
# Set up NVIDIA Cosmos 3 on a DGX Spark. Idempotent; safe to re-run.
#
# Nothing is vendored and nothing is built from source: Cosmos 3 is served through
# diffusers, so this is a venv, a cu130 torch, and a pip install. The model weights
# are pulled separately by `fetch.sh` (or on first use), because a 35 GB download is
# not something a setup script should do behind your back.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

echo "==> venv"
if [ ! -d "$HERE/.venv" ]; then
  uv venv --python 3.12 "$HERE/.venv"
fi
PY="$HERE/.venv/bin/python"

# torch FIRST and from the cu130 index. Order matters: if a later resolve pulls
# plain-PyPI torch it will happily install the CPU-only aarch64 wheel over the top,
# and the only symptom is torch.cuda.is_available() == False much later.
echo "==> torch (cu130, aarch64)"
uv pip install --python "$PY" --index-url "$TORCH_INDEX" torch torchvision

echo "==> cosmos stack"
uv pip install --python "$PY" -r "$HERE/requirements.txt"

echo "==> flyte"
# connectrpc pinned to 0.10.x: 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
uv pip install --python "$PY" "flyte==2.2.1" "connectrpc==0.10.*"

echo "==> verify"
"$PY" - <<'PY'
import torch
print(f"  torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"  device: {torch.cuda.get_device_name(0)}  free {free/2**30:.1f} / {total/2**30:.1f} GiB")
    print(f"  capability: sm_{''.join(str(c) for c in torch.cuda.get_device_capability(0))}")

import diffusers, transformers
print(f"  diffusers {diffusers.__version__}  transformers {transformers.__version__}")

# The three imports that actually matter. If Cosmos3OmniPipeline is missing, the
# diffusers version is too old and nothing else in this repo will work.
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition  # noqa: F401
print("  Cosmos3OmniPipeline imports")
import av  # noqa: F401
print("  PyAV imports")
PY

cat <<EOF

Done. Next:

  ./fetch.sh                       # pull nvidia/Cosmos3-Nano (~35 GB) once
  ./.venv/bin/python smoke_test.py # one 480p clip on the host, no Flyte

Then the Flyte runs:

  ./.venv/bin/flyte run pipeline.py imagine
  ./.venv/bin/flyte run pipeline.py rollout

Before a long session, two host-level knobs that are not optional on this box:

  sudo swapoff -a                 swap thrash on unified memory is a silent freeze
  sudo nvidia-smi -lgc 300,2100   clamp clocks; the hard crashes here are power
                                  spikes (OCP), not OOM
EOF
