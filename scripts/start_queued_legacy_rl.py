"""Start rho=1 immediately; pause only the old queue owner, never its trainers."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def live(pid):
    status = Path(f"/proc/{pid}/stat")
    return status.exists() and status.read_text().split(") ", 1)[1].split()[0] != "Z"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = args.root.resolve()
    marker = out / "parallel-override.json"
    if marker.exists():
        raise RuntimeError("parallel override already exists; inspect before retrying")
    owner = json.loads((out / "launcher.json").read_text())["pid"]
    first = json.loads((out / "rl-rho0.75-mixed-process.json").read_text())
    second = json.loads((out / "rl-rho0.875-mixed-process.json").read_text())
    third_meta = out / "rl-rho1-mixed-process.json"
    if third_meta.exists():
        raise RuntimeError("rho=1 already dispatched")
    assert live(owner) and live(first["pid"]) and live(second["pid"])
    state = dict(
        supervisor_pid=os.getpid(),
        queue_owner_pid=owner,
        status="preparing",
        scope="three trainers concurrent; evaluation lock unchanged",
    )

    def save():
        temp = marker.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(marker)

    save()
    os.kill(owner, signal.SIGSTOP)
    child = None
    try:
        # The stopped owner cannot dequeue rho=1 while this independent trainer runs.
        cmd = [
            arg.replace("scenario-rho0.75.json", "scenario-rho1.json").replace(
                "rl-rho0.75-mixed", "rl-rho1-mixed"
            )
            for arg in first["args"]
        ]
        assert not third_meta.exists()
        with (out / "rl-rho1-mixed.log").open("a") as log:
            child = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    BC_EVAL_LOCK=str(out / "evaluation.lock"),
                    ECSAI_RUN_NAME="MAPPO-night-mixed-rho1-seed0",
                ),
            )
        third_meta.write_text(json.dumps(dict(pid=child.pid, args=cmd)))
        state.update(status="running", trainer_pids=[first["pid"], second["pid"], child.pid])
        save()
        print("STARTED rho=1", child.pid, flush=True)
        while True:
            all_done = True
            for load, pid in zip([0.75, 0.875, 1], state["trainer_pids"], strict=True):
                active = live(pid)
                final = out / f"rl-rho{load:g}-mixed/evaluation-stochastic-262144.json"
                if not active and not final.exists():
                    raise RuntimeError(
                        f"rho={load} exited without final evaluation; queue remains paused"
                    )
                if active:
                    all_done = False
            if all_done:
                break
            time.sleep(15)
        if child.wait() != 0:
            raise RuntimeError("rho=1 trainer failed; queue remains paused")
        # Existing owner reaps its completed children and skips rho=1 after seeing last.pt.
        os.kill(owner, signal.SIGCONT)
        state["status"] = "completed_queue_resumed"
        save()
        print("All trainers finished; original owner resumed for final aggregation", flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error))
        save()
        if child is None:
            os.kill(owner, signal.SIGCONT)
        raise


if __name__ == "__main__":
    main()
