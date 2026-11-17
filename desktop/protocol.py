"""Qt-free worker protocol: command construction and job envelopes."""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def build_worker_command(task, job_path, result_path, frozen=False, root=ROOT):
    if frozen:
        return sys.executable, ["--worker", task, "--job", str(job_path), "--result", str(result_path)]
    return sys.executable, [str(Path(root) / "desktop" / "qt_app.py"),
                            "--worker", task, "--job", str(job_path), "--result", str(result_path)]


def make_job(task: str, payload: dict, task_id: str, created=None) -> dict:
    return {"schema_version": SCHEMA_VERSION, "task_id": task_id,
            "task": task, "created": time.time() if created is None else created,
            "payload": payload}


def parse_job(doc: dict, task: str) -> dict:
    if not isinstance(doc, dict):
        raise ValueError("job must be a JSON object")
    if "payload" in doc:
        schema = doc.get("schema_version")
        if schema not in (None, SCHEMA_VERSION):
            raise ValueError(f"unsupported schema_version {schema}")
        task_in_doc = doc.get("task")
        if task_in_doc and task_in_doc != task:
            raise ValueError(f"job task {task_in_doc!r} != CLI task {task!r}")
        return doc["payload"]
    return doc

