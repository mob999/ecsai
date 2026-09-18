"""Run: uv run python -m examples.dag_fork_join."""

from .common import run_example
from .scenarios import fork_join


def main() -> None:
    result = run_example(fork_join(), "dag-fork-join")
    stages = {stage.stage_id: stage for stage in result.state.stages}
    for predecessor in ("left", "right"):
        assert stages[predecessor].completed_s is not None
        assert stages["join"].started_s is not None
        assert stages["join"].started_s >= stages[predecessor].completed_s


if __name__ == "__main__":
    main()
