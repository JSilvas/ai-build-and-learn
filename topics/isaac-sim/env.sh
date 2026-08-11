# Source this before anything Isaac: `source env.sh`
#
# Nothing here builds or downloads. It only points the shell at the two trees
# that setup.sh created, and fixes the one environment variable that Isaac Sim
# cannot start without on aarch64.

ISAAC_ROOT="${ISAAC_ROOT:-$HOME/isaac}"

export ISAACSIM_PATH="${ISAAC_ROOT}/IsaacSim/_build/linux-aarch64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
export ISAACLAB_PATH="${ISAAC_ROOT}/IsaacLab"

# The libgomp dance, and it is load-bearing on arm64.
#
# Isaac Sim's native extensions link against the SYSTEM OpenMP runtime. PyTorch's
# aarch64 wheels ship their OWN libgomp inside the wheel, and whichever one the
# loader sees first wins for the whole process. Get it backwards and Kit dies
# during extension startup with a symbol-lookup error out of libgomp, usually
# after ~20 seconds of looking like it worked.
#
# APPEND, never assign blind: unset first so a stale value from a previous
# `source env.sh` cannot stack up into a duplicated preload list.
unset LD_PRELOAD
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"

# git-lfs lives in ~/.local/bin on this box (installed as a user binary, since
# the apt package needs root and the build only needs the binary on PATH).
case ":${PATH}:" in
    *":${HOME}/.local/bin:"*) ;;
    *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac
