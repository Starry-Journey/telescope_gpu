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
from desktop.protocol import build_worker_command, make_job

ROOT = Path(__file__).resolve().parents[1]


class TaskRunner(QObject):
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    started = Signal(str)

    def __init__(self, task_root: Path, parent=None, command_builder=None):
        super().__init__(parent)
        self.task_root = Path(task_root)
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.task = None
        self.last_elapsed = 0.0
        self._command_builder = command_builder or build_worker_command
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
        try:
            job.write_text(json.dumps(make_job(task, payload, task_id), default=str),
                           encoding="utf-8")
        except Exception:
            try:
                shutil.rmtree(td)
            except OSError:
                pass
            raise
        proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT))
        self.task = {"id": task_id, "dir": td, "job": job, "result": result,
                     "proc": proc, "cancelled": False, "done": False,
                     "out": bytearray(), "start": time.monotonic()}
        proc.readyReadStandardOutput.connect(lambda: self._read(proc))
        proc.readyReadStandardError.connect(lambda: self._read(proc))
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        prog, args = self._command_builder(task, job, result,
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
        elapsed = time.monotonic() - t["start"]
        self.last_elapsed = elapsed
        return t, raw, data, elapsed

    def _on_finished(self, code, status):
        t = self.task
        if t is None or t["done"]:
            return
        t["done"] = True
        cancelled = bool(t["cancelled"])
        t, raw, data, _elapsed = self._take_result()
        if t is None:
            return
        if cancelled:
            self.cancelled.emit()
            return
        if code != 0 or data is None or (isinstance(data, dict) and "error" in data):
            msg = (data or {}).get("error") if isinstance(data, dict) else None
            msg = msg or ("worker produced no result" if data is None else raw[-2000:] or "worker failed")
            self.failed.emit(str(msg) + "\n\nworker stderr/stdout:\n" + raw[-3000:])
            return
        self.finished.emit(data or {})

    def _on_error(self, err):
        t = self.task
        if t is None or t["done"]:
            return
        t["done"] = True
        cancelled = bool(t["cancelled"])
        t, raw, _data, _elapsed = self._take_result()
        if t is None:
            return
        if cancelled:
            self.cancelled.emit()
            return
        self.failed.emit("worker process error: %s\n%s" % (err, raw[-2000:]))

    def cancel(self):
        t = self.task
        if t is None:
            return False
        t["cancelled"] = True
        proc = t["proc"]
        proc.kill()
        finished = proc.waitForFinished(1500)
        if not finished:
            # forced cancel: kill again so the worker cannot outlive the UI
            proc.kill()
            proc.waitForFinished(500)
        if self.task is t and not t["done"]:
            # settle deterministically: QProcess occasionally swallows the
            # finished/error signals around kill(); late signals are ignored
            # via the done flag / task slot.
            t["done"] = True
            self._take_result()
            self.cancelled.emit()
        return True
