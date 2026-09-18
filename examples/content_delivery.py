"""Run: uv run python -m examples.content_delivery."""

from .common import run_example
from .scenarios import content_delivery


def main() -> None:
    run_example(content_delivery(), "content-delivery")


if __name__ == "__main__":
    main()
