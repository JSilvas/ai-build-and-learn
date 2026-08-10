#!/usr/bin/env bash
# Pull the Cosmos 3 weights to the shared HF cache. Resumable; safe to re-run.
#
# Split out of setup.sh on purpose. This is ~35 GB for Cosmos3-Nano and the two
# things that go wrong here are both slow failures rather than errors:
#
#   * The Spark's HF downloads stall. hf_transfer is a Rust downloader that does its
#     own DNS and ignores socket timeouts, so a black-holed route to the CDN hangs
#     forever rather than erroring. This turns it OFF and sets HF_HUB_DOWNLOAD_TIMEOUT,
#     which bounds a *stalled read* (per-read, not total), so a hung socket fails in
#     ~60s and snapshot_download resumes from the .incomplete file.
#
#   * Disk. A large fetch onto a nearly-full root filesystem evicts every pod on the
#     Flyte cluster, control plane included, and that recovery is much longer than
#     this download. The guard below refuses to start below 60 GB free.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"
REPO="${1:-nvidia/Cosmos3-Nano}"

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60

FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 60 ]; then
  echo "Refusing to fetch: only ${FREE_GB}G free on /." >&2
  echo "A big model pull here will disk-evict the Flyte cluster. Free space first:" >&2
  echo "  docker volume prune -f" >&2
  echo "  kubectl rollout restart deploy/rustfs -n flyte   # also reclaims leaked heap" >&2
  exit 1
fi
echo "==> ${FREE_GB}G free, fetching $REPO"

"$PY" - "$REPO" <<'PY'
import sys, time
from huggingface_hub import snapshot_download

repo = sys.argv[1]
t0 = time.monotonic()
# max_workers=4 rather than the default 8: fetches on this box are more reliable
# serialized than parallel, and a stalled worker is what produces a "download" that
# sits at the same byte count for an hour.
path = snapshot_download(repo_id=repo, max_workers=4)
print(f"{repo} -> {path}  ({(time.monotonic() - t0) / 60:.1f} min)")
PY
