"""Worker QThread for short in-process jobs (e.g. the demo tab render).

Split from desktop/gui.py (old lines 129-141).  Lifecycle contract used by
AppShell._run / AppShell._shutdown_workers:

* run() executes ``self.fn()`` and emits ``result`` or ``failed`` exactly once;
* the owner registers the instance in ``QtApp._workers`` and points
  ``QtApp.active_worker`` at it while it runs;
* at window close the slots are disconnected *before* waiting, so a late
  result can never touch widgets that are already destroyed.
"""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.result.emit(self.fn())
        except Exception as e:
            self.failed.emit(str(e) + "\n\n" + traceback.format_exc())
