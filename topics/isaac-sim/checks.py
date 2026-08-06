"""The four physics checks, and the runner that drives them. Library AND script.

Two callers share this file and must not drift apart: if the containerised run and
the bare-metal run measure different things, comparing them tells you nothing, which
defeats the point of running it in both places.

  bare metal   smoke_test.py is a shim over main() below
  in a pod     pipeline.py's Flyte task spawns THIS FILE as a child process

Being importable is load-bearing for the second one. Flyte only bundles modules that
the task module imports at top level, and `smoke_test.py` cannot be imported at all
(it constructs a SimulationApp at import time). pipeline.py imports `checks`, so this
file rides along into the pod and can be spawned there by path.

Note the isaacsim imports live INSIDE the functions. That is not a style choice: Kit
has to boot before isaacsim.core exists as importable Python. Hoisting them to module
top is the single most common way to break an Isaac Sim script.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

DT = 1.0 / 60.0
GRAVITY = 9.81


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def physics_checks(steps: int = 240, drop_height: float = 2.0) -> list[Check]:
    """Boot a scene, drop a cube, and report what the box actually did.

    Returns one Check per property. Never raises: a failure is a failed Check, so a
    caller always gets a full report rather than a traceback and three unknowns.
    """
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, GroundPlane

    out: list[Check] = []

    try:
        # device="cuda" is the whole point of Isaac Sim over a CPU simulator. If PhysX
        # falls back to CPU it does NOT raise; it just runs, quietly, ~50x slower, and
        # every parallel-env benchmark taken afterwards is then meaningless.
        world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT, device="cuda")
        world.scene.add(GroundPlane(prim_path="/World/ground", size=50.0))

        # np.array, not a list. These kwargs reach USD via .tolist(), so a plain list
        # dies with "'list' object has no attribute 'tolist'" inside PreviewSurface.
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/cube",
                name="cube",
                position=np.array([0.0, 0.0, drop_height]),
                scale=np.array([0.2, 0.2, 0.2]),
                color=np.array([0.2, 0.6, 1.0]),
            )
        )
        world.reset()

        out.append(Check("physx on gpu", "cuda" in str(world.device).lower(), f"physics device = {world.device}"))

        start_z = float(cube.get_world_pose()[0][2])

        # Warm up before timing. The first steps pay for CUDA context creation and
        # PhysX GPU buffer allocation, which is seconds, not milliseconds; timing them
        # would report ~40 steps/sec for a box that does hundreds.
        for _ in range(10):
            world.step(render=False)

        t0 = time.perf_counter()
        for _ in range(steps):
            world.step(render=False)
        elapsed = time.perf_counter() - t0

        dropped = start_z - float(cube.get_world_pose()[0][2])

        # Free fall covers 0.5*g*t^2, but the cube lands and stops, so the real bound
        # is "it fell, and no further than free fall allows". Outside that means
        # gravity is off, the collider is missing, or it tunnelled through the ground.
        free_fall = 0.5 * GRAVITY * (steps * DT) ** 2
        out.append(
            Check(
                "gravity is real",
                0.05 < dropped <= free_fall + 0.1,
                f"cube fell {dropped:.3f} m from {start_z:.2f} m in {steps} steps "
                f"(free fall bound {min(free_fall, start_z):.3f} m)",
            )
        )
        out.append(Check("throughput", elapsed > 0, f"{steps / elapsed:,.0f} physics steps/sec ({elapsed:.2f}s)"))
        world.stop()
    except Exception as exc:  # noqa: BLE001 - a failure is a failed check, not a crash
        out.append(Check(type(exc).__name__, False, str(exc)))

    return out


def gpu_banner() -> str:
    """What Warp thinks the GPU is. Useful in a pod, where you cannot just run nvidia-smi."""
    try:
        import warp as wp

        wp.init()
        dev = wp.get_device("cuda:0")
        return f"{dev.name} | arch sm_{dev.arch} | {dev.total_memory / 1024**3:.0f} GiB"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__}: {exc})"


def main(argv: list[str] | None = None) -> None:
    """Boot Kit, run the checks, print them, optionally dump JSON, exit 0/1.

    Called two ways and it must behave identically in both: directly by
    `smoke_test.py` on the host, and as a child process spawned by pipeline.py's
    Flyte task inside the pod.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Isaac Sim smoke test")
    parser.add_argument("--steps", type=int, default=240, help="physics steps to run (240 = 4s at 60Hz)")
    parser.add_argument("--drop_height", type=float, default=2.0, help="metres to drop the cube from")
    parser.add_argument("--gui", action="store_true", help="open the viewport (needs a display)")
    # stdout is not usable as a result channel: Kit writes hundreds of extension
    # startup lines to it before this prints anything. The Flyte task reads this file.
    parser.add_argument("--json", metavar="PATH", help="also write results as JSON to PATH")
    args = parser.parse_args(argv)

    # SimulationApp boots the Omniverse Kit runtime, and nothing from isaacsim.core
    # can be imported before it. Hence the import here rather than at module top.
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": not args.gui})

    print("\n=== Isaac Sim smoke test ===\n", flush=True)
    gpu = gpu_banner()
    print(f"  gpu: {gpu}\n", flush=True)

    results = [Check("kit boots", True, "SimulationApp came up " + ("with GUI" if args.gui else "headless"))]
    results += physics_checks(steps=args.steps, drop_height=args.drop_height)

    for c in results:
        print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}", flush=True)

    failed = [c.name for c in results if not c.ok]
    print(f"\n=== {len(results) - len(failed)}/{len(results)} checks passed ===", flush=True)
    if failed:
        print("failed: " + ", ".join(failed), flush=True)
    print(flush=True)

    if args.json:
        import json

        with open(args.json, "w") as fh:
            json.dump(
                {
                    "gpu": gpu,
                    "passed": len(results) - len(failed),
                    "total": len(results),
                    "failed": failed,
                    "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in results],
                },
                fh,
            )

    # Everything above happens BEFORE close(), and that ordering is not optional.
    #
    # Kit runs with --/app/fastShutdown=True, so close() calls os._exit() internally
    # and NEVER RETURNS. Anything after it is dead code that silently does not run.
    # Two earlier versions of this put the summary there and printed nothing at all,
    # one of them while a check was failing.
    #
    # close(exit_code=...) is also the only way to exit nonzero: a bare exception or
    # SystemExit gets swallowed by Kit's shutdown path and the process still reports
    # success, which in a smoke test is worse than useless.
    #
    # Keeping fast_shutdown at its default is deliberate. The graceful path
    # (fast_shutdown=False) cancels every asyncio task in the process, which is fine
    # for a standalone script and catastrophic inside Flyte's runtime. Running this as
    # a CHILD process is what lets the pod keep the fast path safely. See pipeline.py.
    sim_app.close(exit_code=1 if failed else 0)


if __name__ == "__main__":
    main()
