#!/usr/bin/env bash
#
# Isaac Sim 6.0 + Isaac Lab on a DGX Spark (GB10, aarch64, DGX OS 7.x / Ubuntu 24.04).
#
# There are no aarch64 wheels for Isaac Sim. Every x86 "pip install isaacsim"
# tutorial is a dead end on this box; the only supported path on Grace-Blackwell
# is a source build, which is what this does.
#
#   ./setup.sh prereqs     apt packages + EULA. NEEDS SUDO, NEEDS A HUMAN.
#   ./setup.sh isaacsim    clone + build Isaac Sim      (~10-15 min compile)
#   ./setup.sh isaaclab    clone + install Isaac Lab    (~10 min of pip)
#   ./setup.sh all         all three, in order
#
# Everything lands in $ISAAC_ROOT (default ~/isaac), NOT in this repo: the built
# tree is ~50 GB and has no business inside git.
#
# Re-running any stage is safe. Each one checks for its own output first.

set -euo pipefail

ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaac}"
ISAACSIM_SRC="${ISAAC_ROOT}/IsaacSim"
ISAACLAB_SRC="${ISAAC_ROOT}/IsaacLab"
ISAACSIM_BUILD="${ISAACSIM_SRC}/_build/linux-aarch64/release"

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!!! %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------

check_host() {
    [ "$(uname -m)" = "aarch64" ] || die "Not aarch64. This script is Spark-only."
    command -v nvidia-smi >/dev/null || die "No nvidia-smi on PATH."

    # 50 GB is NVIDIA's number for the build artifacts. Ask for 80 GB: the Spark
    # shares one NVMe with the Flyte blob store and the docker registry, and a
    # disk that fills mid-build takes the whole k8s control plane down with it
    # (kubelet evicts on DiskPressure, and it does not care what you were doing).
    local avail_gb
    avail_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
    [ "${avail_gb}" -ge 80 ] || die "Only ${avail_gb} GB free on \$HOME. Want 80+. Free space first."
    say "host ok: aarch64, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${avail_gb} GB free"
}

# ---------------------------------------------------------------------------

stage_prereqs() {
    check_host

    # gcc-11 specifically. Noble defaults to gcc-13 and Isaac Sim's Kit SDK will
    # not compile against it; gcc-11 is still in noble-updates/universe, so this
    # needs no PPA. update-alternatives at priority 200 makes 11 the default
    # while leaving 13 installed and selectable.
    say "apt: gcc-11, git-lfs, X11/GL headers (Isaac Lab links against these)"
    sudo apt-get update
    sudo apt-get install -y \
        gcc-11 g++-11 git git-lfs build-essential curl \
        python3.12-dev \
        libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev

    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 200
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 200
    say "gcc is now: $(gcc --version | head -1)"

    # The EULA is a real license agreement for the Omniverse Kit SDK and the
    # bundled 3D assets. It is deliberately interactive and deliberately not
    # scripted around: a human accepts it, or the build does not happen. It lives
    # in the source tree, so the clone has to come first.
    clone_isaacsim
    if [ -f "${ISAACSIM_SRC}/.eula_accepted" ]; then
        say "EULA already accepted"
    else
        say "NVIDIA Omniverse license. Read it, answer honestly."
        ( cd "${ISAACSIM_SRC}" && ./tools/eula_check.sh ) || die "EULA declined. Nothing further to do."
    fi
}

# ---------------------------------------------------------------------------

clone_isaacsim() {
    mkdir -p "${ISAAC_ROOT}"
    if [ -d "${ISAACSIM_SRC}/.git" ]; then
        say "Isaac Sim source already at ${ISAACSIM_SRC}, skipping clone"
        return
    fi
    say "cloning Isaac Sim (shallow + submodules + LFS, ~1.3 GB)"
    git clone --depth=1 --recursive https://github.com/isaac-sim/IsaacSim "${ISAACSIM_SRC}"
    ( cd "${ISAACSIM_SRC}" && git lfs install && git lfs pull )
}

stage_isaacsim() {
    check_host
    clone_isaacsim

    [ -f "${ISAACSIM_SRC}/.eula_accepted" ] || die "EULA not accepted. Run: ./setup.sh prereqs"
    case "$(gcc --version | head -1)" in
        *' 11.'*) ;;
        *) die "gcc is $(gcc --version | head -1), need 11.x. Run: ./setup.sh prereqs" ;;
    esac

    if [ -x "${ISAACSIM_BUILD}/isaac-sim.sh" ]; then
        say "already built at ${ISAACSIM_BUILD} (rm -rf _build to force a rebuild)"
        return
    fi

    say "building Isaac Sim. 10-15 min; it pulls several GB of Kit SDK packages first."
    ( cd "${ISAACSIM_SRC}" && ./build.sh )
    [ -x "${ISAACSIM_BUILD}/isaac-sim.sh" ] || die "build.sh finished but ${ISAACSIM_BUILD}/isaac-sim.sh is missing."
    say "built: ${ISAACSIM_BUILD}"
}

# ---------------------------------------------------------------------------

stage_isaaclab() {
    [ -x "${ISAACSIM_BUILD}/isaac-sim.sh" ] || die "Isaac Sim is not built. Run: ./setup.sh isaacsim"

    if [ -d "${ISAACLAB_SRC}/.git" ]; then
        say "Isaac Lab source already at ${ISAACLAB_SRC}, skipping clone"
    else
        say "cloning Isaac Lab"
        git clone --recursive https://github.com/isaac-sim/IsaacLab "${ISAACLAB_SRC}"
    fi

    # Isaac Lab finds its simulator through a symlink named _isaac_sim, and only
    # through that symlink. -n so a re-run replaces the link instead of nesting
    # a second one inside the directory it already points at.
    ln -sfn "${ISAACSIM_BUILD}" "${ISAACLAB_SRC}/_isaac_sim"

    say "installing Isaac Lab + RL libs (rsl_rl, rl_games, skrl, sb3) into Isaac Sim's python"
    ( cd "${ISAACLAB_SRC}" && ./isaaclab.sh --install )
    say "installed. ${ISAACLAB_SRC}"
}

# ---------------------------------------------------------------------------

case "${1:-all}" in
    prereqs)  stage_prereqs ;;
    isaacsim) stage_isaacsim ;;
    isaaclab) stage_isaaclab ;;
    all)      stage_prereqs; stage_isaacsim; stage_isaaclab ;;
    *)        die "usage: $0 {prereqs|isaacsim|isaaclab|all}" ;;
esac

say "done. Next: source env.sh && \$ISAACSIM_PYTHON_EXE smoke_test.py"
