"""Versioned analysis files, with no RL-specific observations or rewards."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from edge_sim_models import CommandRecord, RunResult


def export_run(
    result: RunResult, directory: str | Path, commands: tuple[CommandRecord, ...] = ()
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    (target / "result.json").write_text(result.model_dump_json(indent=2))
    if result.manifest is not None:
        (target / "manifest.json").write_text(result.manifest.model_dump_json(indent=2))
    metadata = {b"edge_sim.schema_version": b"1", b"edge_sim.kind": b"domain_events"}
    schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("sequence", pa.int64()),
            ("time_s", pa.float64()),
            ("kind", pa.string()),
            ("entity_id", pa.string()),
            ("details_json", pa.string()),
        ],
        metadata=metadata,
    )
    rows = [
        dict(
            run_id=result.run_id,
            sequence=e.sequence,
            time_s=e.time_s,
            kind=e.kind,
            entity_id=e.entity_id,
            details_json=json.dumps({d.name: d.value for d in e.details}, sort_keys=True),
        )
        for e in result.events
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), target / "events.parquet")
    command_schema = pa.schema(
        [
            ("run_id", pa.string()),
            ("decision_id", pa.string()),
            ("revision", pa.int64()),
            ("time_s", pa.float64()),
            ("commands_json", pa.string()),
            ("view_json", pa.string()),
            ("policy_id", pa.string()),
        ],
        metadata={b"edge_sim.schema_version": b"1", b"edge_sim.kind": b"decisions"},
    )
    command_rows = [
        {
            "run_id": row["run_id"],
            "decision_id": row["decision_id"],
            "revision": row["revision"],
            "time_s": row["time_s"],
            "commands_json": json.dumps(row["commands"], sort_keys=True),
            "view_json": json.dumps(row.get("view"), sort_keys=True),
            "policy_id": row.get("policy_id", "external"),
        }
        for record in commands
        for row in (record.model_dump(mode="json"),)
    ]
    pq.write_table(
        pa.Table.from_pylist(command_rows, schema=command_schema), target / "decisions.parquet"
    )
    return target
