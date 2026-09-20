"""Dependency-free SVG overview of the long-form training metrics CSV."""

import argparse
import csv
from html import escape
from pathlib import Path


def plot(folder):
    folder = Path(folder)
    rows = list(csv.DictReader((folder / "metrics.csv").open()))
    panels = [
        ("Reward", ["train/reward"]),
        ("Validation success rate", ["eval/success_rate"]),
        (
            "Backhaul allocation",
            sorted({r["metric"] for r in rows if "/action_4_mean" in r["metric"]}),
        ),
        ("Actor KL", sorted({r["metric"] for r in rows if r["metric"].endswith("_kl")})),
    ]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="700" viewBox="0 0 1100 700">',
        '<rect width="1100" height="700" fill="white"/>',
        "<style>text{font:13px sans-serif;fill:#333} .title{font-size:18px}</style>",
    ]
    colors = ["#1765ad", "#df701b", "#1a8b60", "#9a489e", "#bd4242", "#337c87", "#656565"]
    for index, (title, keys) in enumerate(panels):
        left, top = 70 + (index % 2) * 550, 45 + (index // 2) * 340
        series = [
            (key, [(float(r["env_steps"]), float(r["value"])) for r in rows if r["metric"] == key])
            for key in keys
        ]
        points = [p for _, values in series for p in values]
        if not points:
            continue
        xmax = max(x for x, _ in points) or 1
        lo, hi = min(y for _, y in points), max(y for _, y in points)
        if index in (1, 2):
            lo, hi = 0, 1
        if hi == lo:
            hi = lo + 1
        svg.append(f'<text class="title" x="{left}" y="{top - 15}">{escape(title)}</text>')
        for j in range(5):
            y = top + j * 52.5
            value = hi - j * (hi - lo) / 4
            svg.append(
                f'<path d="M{left},{y} h430" stroke="#ddd"/>'
                f'<text x="{left - 60}" y="{y + 4}">{value:.3g}</text>'
            )
        for j, (key, values) in enumerate(series):
            coords = " ".join(
                f"{left + x / xmax * 430:.2f},{top + (hi - y) / (hi - lo) * 210:.2f}"
                for x, y in values
            )
            color = colors[j % len(colors)]
            svg.append(
                f'<polyline points="{coords}" stroke="{color}" fill="none" stroke-width="2"/>'
            )
            if len(values) == 1:
                x, y = coords.split(",")
                svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
            label = key.removeprefix("train/").removeprefix("eval/")
            svg.append(
                f'<text x="{left}" y="{top + 250 + j * 15}" style="fill:{color}">'
                f"{escape(label)}</text>"
            )
        svg.append(
            f'<text x="{left}" y="{top + 230}">0</text>'
            f'<text x="{left + 310}" y="{top + 230}">{xmax:g} env steps</text>'
        )
    svg.append("</svg>")
    (folder / "curves.svg").write_text("\n".join(svg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("folder")
    plot(parser.parse_args().folder)
