"""Offline HTML reports; rendering never imports or starts the simulation engine."""

import json
from importlib.resources import files
from pathlib import Path

from edge_sim_models import CommandRecord, RunManifest, RunResult, RunSpec, ScenarioSpec
from pydantic import BaseModel, ConfigDict, TypeAdapter


class VisualizationData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "edge-sim-visualization-v1"
    scenario: ScenarioSpec | None = None
    result: RunResult | None = None
    run: RunSpec | None = None
    commands: tuple[CommandRecord, ...] = ()


def visualization_data(
    source: ScenarioSpec | RunSpec | RunResult | RunManifest,
    commands: tuple[CommandRecord, ...] = (),
) -> VisualizationData:
    result = source if isinstance(source, RunResult) else None
    run = source if isinstance(source, RunSpec) else None
    if isinstance(source, RunManifest):
        run = source.run_spec
        if run is None:
            raise ValueError("Manifest has no run_spec; provide a scenario or result JSON")
    if result is not None and result.manifest is not None:
        run = result.manifest.run_spec
    scenario = source if isinstance(source, ScenarioSpec) else run.scenario if run else None
    if commands and (result is None or any(c.run_id != result.run_id for c in commands)):
        raise ValueError("Command records must belong to the visualized result's run_id")
    return VisualizationData(scenario=scenario, run=run, result=result, commands=commands)


def export_html(
    source: ScenarioSpec | RunSpec | RunResult | RunManifest,
    destination: str | Path,
    commands: tuple[CommandRecord, ...] = (),
) -> Path:
    """Write one self-contained HTML file with safely embedded, validated data."""
    data = visualization_data(source, commands)
    # Script raw-text parsing recognizes closing tags even inside JSON strings.
    payload = data.model_dump_json().replace("&", "\\u0026").replace("<", "\\u003c")
    payload = payload.replace(">", "\\u003e").replace("\u2028", "\\u2028")
    payload = payload.replace("\u2029", "\\u2029")
    template = files("edge_sim").joinpath("templates/report.html").read_text(encoding="utf-8")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__EDGE_SIM_PAYLOAD__", payload), encoding="utf-8")
    return target


def visualize_file(
    source: Path, destination: Path, commands_path: Path | None = None
) -> Path:
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected a scenario, run, result, or manifest JSON object")
    if "state" in raw:
        model = RunResult.model_validate(raw)
    elif "nodes" in raw:
        model = ScenarioSpec.model_validate(raw)
    elif "scenario" in raw:
        model = RunSpec.model_validate(raw)
    elif "run_spec" in raw:
        model = RunManifest.model_validate(raw)
    else:
        raise ValueError("Expected a scenario, run, result, or manifest JSON object")
    commands = (
        TypeAdapter(tuple[CommandRecord, ...]).validate_json(commands_path.read_text("utf-8"))
        if commands_path
        else ()
    )
    if destination.resolve() in {p.resolve() for p in (source, commands_path) if p}:
        raise ValueError("HTML output must not overwrite an input file")
    return export_html(model, destination, commands)
