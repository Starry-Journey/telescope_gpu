"""Per-task QProcess runner.

Each task owns its own cancelled flag, QProcess, temp directory and result
path.  Exit callbacks only touch their own task, so a late old process cannot
clear a new task's state.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

ROOT = Path(__file__).resolve().parents[1]


def build_worker_command(task, job_path, result_path, frozen=False, root=ROOT):
    """Return (program, args) for a worker; source mode must use qt_app.py."""
    if frozen:
        return sys.executable, ["--worker", task, "--job", str(job_path), "--result", str(result_path)]
    return sys.executable, [str(Path(root) / "desktop" / "qt_app.py"),
                            "--worker", task, "--job", str(job_path), "--result", str(result_path)]


class TaskRunner(QObject):
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    started = Signal(str)

    def __init__(self, task_root: Path, parent=None):
        super().__init__(parent)
        self.task_root = Path(task_root)
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.task = None
        self._cleanup_stale()

    def _cleanup_stale(self):
        for d in self.task_root.glob("task_*"):
            try:
                shutil.rmtree(d)
            except OSError:
                pass

    def busy(self) -> bool:
        return self.task is not None

    def start(self, task: str, payload: dict):
        if self.task is not None:
            raise RuntimeError("a task is already running")
        task_id = uuid.uuid4().hex[:12]
        td = Path(tempfile.mkdtemp(prefix="task_", dir=str(self.task_root)))
        job = td / "job.json"
        result = td / "result.json"
        job.write_text(json.dumps({"schema_version": 1, "task_id": task_id,
                                   "task": task, "created": time.time(),
                                   "payload": payload}, default=str), encoding="utf-8")
        proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT))
        self.task = {"id": task_id, "dir": td, "job": job, "result": result,
                     "proc": proc, "cancelled": False, "done": False,
                     "out": bytearray(), "start": time.time()}
        proc.readyReadStandardOutput.connect(lambda: self._read(proc))
        proc.readyReadStandardError.connect(lambda: self._read(proc))
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        prog, args = build_worker_command(task, job, result,
                                          frozen=getattr(sys, "frozen", False))
        proc.start(prog, args)
        self.started.emit(task_id)

    def _read(self, proc):
        if self.task and self.task["proc"] is proc:
            self.task["out"].extend(bytes(proc.readAllStandardOutput()))
            self.task["out"].extend(bytes(proc.readAllStandardError()))

    def _take_result(self):
        t = self.task
        if t is None:
            return None, "", None, 0.0
        self.task = None
        raw = bytes(t["out"]).decode("utf-8", "replace").strip()
        data = None
        try:
            if t["result"].exists():
                data = json.loads(t["result"].read_text(encoding="utf-8"))
        except Exception as e:
            data = {"error": str(e)}
        try:
            shutil.rmtree(t["dir"])
        except OSError:
            pass
        return t, raw, data, time.time() - t["start"]

    def _on_finished(self, code, status):
        t = self.task
        if t is None or t["done"]:
            return
        t["done"] = True
        cancelled = bool(t["cancelled"])
        t, raw, data, elapsed = self._take_result()
        if t is None:
            return
        if cancelled:
            self.cancelled.emit()
            return
        if code != 0 or (isinstance(data, dict) and "error" in data):
            msg = (data or {}).get("error", raw[-2000:] or "worker failed")
            self.failed.emit(str(msg) + "\n\nworker stderr/stdout:\n" + raw[-3000:])
            return
        self.finished.emit(data or {})

    def _on_error(self, err):
        t = self.task
        if t is None or t["done"]:
            return
        t["done"] = True
        t, raw, data, elapsed = self._take_result()
        if t is None:
            return
        self.failed.emit("worker process error: %s\n%s" % (err, raw[-2000:]))

    def cancel(self):
        t = self.task
        if t is None:
            return False
        t["cancelled"] = True
        t["proc"].kill()
        t["proc"].waitForFinished(1500)
        return True
