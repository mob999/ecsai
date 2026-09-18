"""Spawn entrypoint: importing this module never initializes SimGrid."""

import tempfile
import traceback
from pathlib import Path


def worker_main(connection, run_spec):
    # One process, engine and controlling actor per run. Blocking on the pipe
    # inside that actor freezes simulated time while the external policy works.
    try:
        import simgrid

        from .platform import platform_xml
        from .resources import CommandError
        from .runtime import Runtime

        engine = simgrid.Engine(["edge-sim", "--log=root.thres:critical"])
        simgrid.Engine.set_config("network/model:raw")
        simgrid.Engine.set_config("cpu/model:Cas01")
        with tempfile.TemporaryDirectory(prefix="edge-sim-") as directory:
            platform = Path(directory) / "platform.xml"
            platform.write_text(platform_xml(run_spec.scenario))
            engine.load_platform(str(platform))

            def controller():
                runtime = Runtime(run_spec)
                connection.send(("ready", None))
                try:
                    while True:
                        op, payload = connection.recv()
                        try:
                            if op == "close":
                                runtime.close()
                                connection.send(("ok", None))
                                break
                            if op == "advance":
                                value = runtime.advance(payload)
                            elif op == "apply":
                                value = runtime.apply(*payload)
                            elif op == "inspect":
                                value = runtime.inspect(payload)
                            elif op == "result":
                                value = runtime.result()
                            elif op == "events":
                                value = tuple(runtime.events)
                            elif op == "commands":
                                value = tuple(runtime.command_log)
                            else:
                                raise CommandError("unknown_operation", f"Unknown operation: {op}")
                            connection.send(("ok", value))
                        except CommandError as error:
                            connection.send(("error", {"code": error.code, "message": str(error)}))
                except EOFError:
                    runtime.close()
                except Exception:
                    connection.send(
                        ("error", {"code": "backend_error", "message": traceback.format_exc()})
                    )
                    runtime.close()

            simgrid.Actor.create(
                "controller", simgrid.Host.by_name(run_spec.scenario.nodes[0].id), controller
            )
            engine.run()
    except Exception:
        try:
            connection.send(("error", {"code": "worker_failed", "message": traceback.format_exc()}))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
