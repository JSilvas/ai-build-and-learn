"""Does Isaac Sim actually work on this Spark? Answer in about 60 seconds, bare metal.

    source env.sh && $ISAACSIM_PYTHON_EXE smoke_test.py

Not a demo. This is the thing you run after `setup.sh` to find out whether the build
is real, and the thing you run again when something later breaks and you need to know
whether the simulator or your own code is at fault.

It checks four things, in the order they tend to fail:

  1. Kit starts headless at all           (libgomp, missing X libs, bad build)
  2. PhysX picked the GB10, not the CPU   (silent 50x slowdown if it did not)
  3. Gravity is real                      (a cube falls the distance calculus says)
  4. How fast this box steps physics      (a number to compare against later)

Exit code is 0 only if all four pass, so it drops straight into a shell script.

This file is a shim on purpose. The runner lives in checks.py because the Flyte task
in pipeline.py spawns that same module as a child process, and the host number and
the containerised number are only comparable if they run identical code. Options are
the same: --steps, --drop_height, --gui, --json PATH.
"""

from checks import main

main()
