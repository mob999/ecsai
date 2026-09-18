"""Offline reports must preserve contracts without importing a native engine."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from edge_sim.visualization import export_html, visualization_data, visualize_file
from edge_sim_models import (
    CommandRecord,
    Defer,
    Detail,
    DomainEvent,
    RunManifest,
    RunResult,
    RunSpec,
)

from examples.scenarios import fork_join


def embedded_data(path):
    html = path.read_text(encoding="utf-8")
    payload = re.search(
        r'<script id="edge-sim-data" type="application/json">(.*?)</script>', html, re.S
    )
    assert payload is not None
    return json.loads(payload[1])


@pytest.fixture
def run():
    return RunSpec(scenario=fork_join(), run_id="visual-demo", seed=42)


@pytest.fixture
def result(run):
    return RunResult(
        run_id=run.run_id,
        seed=run.seed,
        manifest=RunManifest(run_id=run.run_id, seed=run.seed, run_spec=run),
    )


@pytest.mark.parametrize("kind", ["scenario", "run", "manifest", "result", "legacy-result"])
def test_supported_inputs_roundtrip(tmp_path, run, result, kind):
    source = {
        "scenario": run.scenario,
        "run": run,
        "manifest": result.manifest,
        "result": result,
        "legacy-result": RunResult(run_id="legacy"),
    }[kind]
    path = tmp_path / "input.json"
    path.write_text(source.model_dump_json(), encoding="utf-8")
    output = visualize_file(path, tmp_path / "nested" / "report.html")
    payload = embedded_data(output)
    assert payload == visualization_data(source).model_dump(mode="json")
    html = output.read_text(encoding="utf-8")
    assert "__EDGE_SIM_" not in html
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert "Stage timeline" in html
    assert "Network topology" in html


def test_script_breakout_and_template_tokens_are_inert(tmp_path, result):
    hostile = "</script><script>globalThis.injected=true</script>&\u2028\u2029__EDGE_SIM_SCRIPT__"
    result = result.model_copy(
        update={
            "events": (
                DomainEvent(
                    time_s=0,
                    kind="test",
                    entity_id=hostile,
                    details=(Detail(name="message", value=hostile),),
                ),
            )
        }
    )
    target = export_html(result, tmp_path / "report.html")
    html = target.read_text(encoding="utf-8")
    assert hostile not in html
    assert html.count("</script>") == 2
    assert embedded_data(target)["result"]["events"][0]["entity_id"] == hostile


def test_commands_preserved_and_wrong_run_rejected(tmp_path, run, result):
    command = CommandRecord(
        run_id=run.run_id,
        decision_id="d1",
        revision=1,
        time_s=0,
        commands=(Defer(until_s=1),),
        policy_id="external",
    )
    source = tmp_path / "result.json"
    source.write_text(result.model_dump_json())
    commands = tmp_path / "commands.json"
    commands.write_text(json.dumps([command.model_dump(mode="json")]))
    output = visualize_file(source, tmp_path / "report.html", commands)
    assert embedded_data(output)["commands"] == [command.model_dump(mode="json")]
    with pytest.raises(ValueError, match="run_id"):
        visualization_data(result, (command.model_copy(update={"run_id": "other"}),))
    with pytest.raises(ValueError, match="run_id"):
        visualization_data(run, (command,))
    for target in (source, commands):
        before = target.read_bytes()
        with pytest.raises(ValueError, match="overwrite"):
            visualize_file(source, target, commands)
        assert target.read_bytes() == before
    alias = tmp_path / "alias.html"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="overwrite"):
        visualize_file(source, alias)


@pytest.mark.parametrize("raw", [[], {}, {"nodes": [{"id": "bad"}]}])
def test_invalid_sources_fail_without_output(tmp_path, raw):
    source, destination = tmp_path / "invalid.json", tmp_path / "report.html"
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        visualize_file(source, destination)
    assert not destination.exists()


def test_manifest_without_configuration_fails():
    with pytest.raises(ValueError, match="no run_spec"):
        visualization_data(RunManifest(run_id="old"))


def test_cli_runs_without_loading_simgrid(tmp_path, run):
    source = tmp_path / "run.json"
    source.write_text(run.model_dump_json())
    script = """
import sys
class NoEngine:
    def find_spec(self, fullname, *args):
        if fullname == 'simgrid' or fullname.startswith('edge_sim_simgrid'):
            raise AssertionError('Visualization imported the engine')
sys.meta_path.insert(0, NoEngine())
from edge_sim.cli import main
main()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, "visualize", str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert Path(completed.stdout.strip()) == source.with_suffix(".html")
    assert embedded_data(source.with_suffix(".html"))["run"]["seed"] == 42
