#!/usr/bin/env bash
# Free the GB10's unified pool before a Cosmos run, and say whether it worked.
#
# There is ONE 119.7 GiB pool on this box shared by the GPU, the OS, the kernel's
# page cache and every pod. Two things eat it between runs:
#
#   1. rustfs, the Flyte object store, leaks anonymous heap. It climbs to tens of GB
#      and never gives it back. That memory is NOT reclaimable, so it is a straight
#      subtraction from what a model can have. A rollout restart reclaims it and the
#      data is safe on the PVC.
#   2. Page cache from the 35 GB checkpoint download. This one is clean and IS
#      reclaimable on demand, so it does not actually block an allocation -- but
#      cuMemGetInfo counts it as taken, so it makes every "how much is free" reading
#      lie low. Dropping it is mostly so the numbers you read are the real ones.
#
# Run this before `flyte run pipeline.py ...` on a box that has been up a while.

set -euo pipefail

RUSTFS_RESTART_GIB=${RUSTFS_RESTART_GIB:-6}

mem() {
    awk -v label="$1" '
        /^MemTotal:/     { total = $2 }
        /^MemAvailable:/ { avail = $2 }
        END { printf "%-8s available %6.1f GiB of %.1f GiB\n", label, avail/1048576, total/1048576 }
    ' /proc/meminfo
}

rustfs_gib() {
    # Sum RSS across every rustfs process; 0 if none are running.
    ps -o rss= -C rustfs 2>/dev/null | awk '{ s += $1 } END { printf "%.1f", s/1048576 }'
}

echo "== before =="
mem "host"
echo "rustfs   holding  $(rustfs_gib) GiB of anonymous heap"

held=$(rustfs_gib)
if awk -v h="$held" -v t="$RUSTFS_RESTART_GIB" 'BEGIN { exit !(h > t) }'; then
    echo
    echo "-- restarting rustfs (${held} GiB > ${RUSTFS_RESTART_GIB} GiB threshold) --"
    # Data lives on the PVC, so this only drops the leaked heap. Wait for the new
    # pod to be Ready before continuing: a Flyte task that starts while the object
    # store is down fails on its first blob write, which looks nothing like a memory
    # problem and wastes a debugging session.
    kubectl rollout restart deploy/rustfs -n flyte
    kubectl rollout status deploy/rustfs -n flyte --timeout=180s
else
    echo "rustfs under the ${RUSTFS_RESTART_GIB} GiB threshold, leaving it alone"
fi

echo
echo "-- dropping page cache (optional; needs passwordless sudo) --"
# -n only, never an interactive prompt: this script gets run from agents and CI where
# a password prompt is an invisible hang rather than a question.
if sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    echo "dropped"
else
    # Not fatal, and usually not even worth doing. Page cache is reclaimable, so the
    # run gets the memory either way; dropping it only makes cuMemGetInfo stop
    # under-reporting. guard_memory() reads MemAvailable precisely so it is not fooled.
    echo "skipped (no passwordless sudo). Page cache is reclaimable, so this is"
    echo "cosmetic. To do it by hand:  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
fi

echo
echo "== after =="
mem "host"
echo "rustfs   holding  $(rustfs_gib) GiB of anonymous heap"

# The number that decides whether the run starts. guard_memory() applies the same
# 8 GiB headroom and refuses below ~46 GiB, so check it here rather than finding out
# after a 35 GB download.
awk '
    /^MemAvailable:/ { avail = $2/1048576 }
    END {
        budget = avail - 8
        printf "\nbudget after 8 GiB headroom: %.1f GiB\n", budget
        if (budget < 46) {
            printf "STILL TOO LOW for Cosmos3-Nano (~46 GiB). Something else is holding the pool:\n"
            printf "  ps aux --sort=-rss | head\n"
            exit 1
        }
        printf "enough for Cosmos3-Nano (~46 GiB). Good to run.\n"
    }
' /proc/meminfo
