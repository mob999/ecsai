"""Reproducible native sampling profile; no Torch imports or mock network.

Run this script against either revision using PYTHONPATH pointing to its package
source directories. Compare semantic_sha256 (all windows, final requests and
events), not just aggregate success counts. Profile and timing are separate runs.
"""

import argparse
import cProfile
import hashlib
import json
import math
import pstats
import tempfile
from pathlib import Path
from time import perf_counter


def compare_semantics(reference, actual):
    """Require identical decisions/order; tolerate native floating-point summation."""
    errors = []
    maximum = {"bytes": 0.0, "seconds": 0.0}

    def visit(a, b, path=""):
        if type(a) is not type(b):
            errors.append(path)
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                errors.append(path + "/keys")
            for key in a.keys() & b.keys():
                visit(a[key], b[key], path + "/" + key)
        elif isinstance(a, list):
            if len(a) != len(b):
                errors.append(path + "/length")
            for i, (x, y) in enumerate(zip(a, b, strict=False)):
                visit(x, y, path + "/" + str(i))
        elif isinstance(a, float):
            kind = "bytes" if "bytes" in path or "byte_seconds" in path else "seconds"
            maximum[kind] = max(maximum[kind], abs(a - b))
            tolerance = 1e-3 if kind == "bytes" else 1e-11
            if not math.isclose(a, b, rel_tol=0, abs_tol=tolerance):
                errors.append(path)
        elif a != b:
            errors.append(path)

    visit(reference, actual)
    return {
        "passed": not errors,
        "mismatches": len(errors),
        "examples": errors[:10],
        "max_absolute_difference": maximum,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "small", "medium", "large"], default="small")
    parser.add_argument("--cycles", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy", choices=["threshold", "random", "local", "forward"], default="local"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument(
        "--compare", type=Path, help="Reference semantics.json from an older revision"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import simgrid as sg
    from edge_sim_learning.scenario import ScenarioConfig, build_run
    from edge_sim_models import SchedulerControl, WindowControl
    from edge_sim_simgrid.content import ContentRuntime
    from edge_sim_simgrid.platform import platform_xml

    config = ScenarioConfig.profile(args.profile).model_copy(update={"cycles": args.cycles})
    run, _ = build_run(config, args.seed)
    run = run.model_copy(update={"trace": True})
    engine = sg.Engine(["sampling-profile", "--log=root.thres:critical"])
    sg.Engine.set_config("network/model:raw")
    sg.Engine.set_config("cpu/model:Cas01")
    with tempfile.TemporaryDirectory() as temp:
        platform = Path(temp) / "platform.xml"
        platform.write_text(platform_xml(run.scenario))
        engine.load_platform(str(platform))

        def controller():
            runtime = ContentRuntime(run)
            windows = []
            elapsed = 0.0
            profiler = cProfile.Profile()
            if args.cprofile:
                profiler.enable()
            for step in range(config.cycles):
                control = WindowControl(
                    policy=args.policy,
                    schedulers=tuple(
                        SchedulerControl(
                            cluster_id=s.id,
                            weights=(1, -2, 3, -2 if step % 2 else 1),
                        )
                        for s in run.content.schedulers
                    ),
                )
                started = perf_counter()
                response = runtime.advance_window((step + 1) * config.period_s, control)
                elapsed += perf_counter() - started
                value = response.model_dump(mode="json", exclude={"simulation_wall_s"})
                value["view"].pop("scope", None)  # Added metadata, not a physical change.
                windows.append(value)
            if args.cprofile:
                profiler.disable()
                profiler.dump_stats(str(args.output / "kernel.prof"))
                with (args.output / "profile.txt").open("w") as stream:
                    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                        "cumulative"
                    ).print_stats(50)
            result = runtime.result()
            semantics = {
                "windows": windows,
                "state": result.state.model_dump(mode="json"),
                "events": [e.model_dump(mode="json") for e in runtime.events],
            }
            encoded = json.dumps(semantics, sort_keys=True, separators=(",", ":"))
            (args.output / "semantics.json").write_text(encoded)
            report = {
                "scenario": config.model_dump(),
                "seed": args.seed,
                "policy": args.policy,
                "profiled": args.cprofile,
                "trace": True,
                "kernel_wall_s": elapsed,
                "env_steps_s": config.cycles / elapsed,
                "semantic_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "arrived": response.view.arrived,
                "completed": response.view.completed,
                "timed_out": response.view.timed_out,
                "rejected": response.view.rejected,
            }
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            if args.compare:
                comparison = compare_semantics(json.loads(args.compare.read_text()), semantics)
                (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2))
                assert comparison["passed"], comparison
            print(json.dumps(report))
            runtime.close()

        sg.Actor.create("controller", sg.Host.by_name(run.scenario.nodes[0].id), controller)
        engine.run()


if __name__ == "__main__":
    main()
