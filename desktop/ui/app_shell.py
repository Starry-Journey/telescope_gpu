"""Window shell of the Qt frontend (mixin): layout, menus, task glue, shutdown.

Split from desktop/gui.py (old lines 199-465) into the AppShell mixin.  It is
the only mixin with an ``__init__``; the per-tab mixins (TabCCD...TabDemo) and
ExportActions carry methods only, so ``QtApp(TabCCD, ..., AppShell,
QMainWindow)`` has a single cooperative __init__ and all cross-tab state is
created here (psf/defocus/last_ccd/last_pd/last_fea/last_fea3d/_fea_xyz/
last_s/frames/_workers/active_worker/runner/_task_done/_task_label/log_path/
badge/cancel_btn/tabs/_settings/...).

Fixes kept in this module (see closeEvent/_run_process):
* closeEvent now detaches + joins the in-process Worker QThreads before the
  window dies, then cancels the QProcess task as before;
* _run_process no longer returns silently when the runner is busy — it shows
  a one-line status-bar hint.
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QSettings, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from desktop.tasks import TaskRunner
from desktop.ui.paths import ROOT
from desktop.ui.theme import BAD, GOOD, WARN
from desktop.ui.worker_thread import Worker


class AppShell:
    """Window shell + task/worker lifecycle (no own Qt base)."""

    def __init__(self, startup_t0=None):
        super().__init__()
        self._startup_t0 = startup_t0
        self.setWindowTitle("Telescope Engineering · Qt")
        self.resize(1440, 940)
        self.psf = None
        self.defocus = None
        self.last_ccd = None
        self.last_pd = None
        self.last_fea = None
        self.last_fea3d = None
        self._fea_xyz = None  # node arrays (mm/mm/nm) backing the 3D view
        self.last_s = None
        self.frames = []
        self._workers = []
        self.active_worker = None
        self.runner = TaskRunner(self._task_root(), self)
        self._task_done = None
        self._task_label = ""
        self.runner.finished.connect(self._on_task_done)
        self.runner.failed.connect(self._on_task_failed)
        self.runner.cancelled.connect(self._on_task_cancelled)
        if getattr(sys, "frozen", False):
            self.log_path = Path(sys.executable).resolve().parent / "qt_gui.log"
        else:
            self.log_path = ROOT / "results" / "qt_gui.log"
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        central = QWidget(); root = QVBoxLayout(central)
        header = QHBoxLayout()
        left = QVBoxLayout()
        title = QLabel("Telescope Engineering"); title.setObjectName("Title")
        sub = QLabel("CCD / 相位差 / FEA / ZMX / 控制 · Qt native")
        sub.setObjectName("Subtitle")
        left.addWidget(title); left.addWidget(sub)
        header.addLayout(left, 1)
        self.badge = QLabel("READY"); self.badge.setStyleSheet(
            "background:#12351f;color:#3fb950;border:1px solid #1f6f3a;border-radius:11px;padding:3px 10px;font-weight:600;")
        header.addWidget(self.badge, 0, Qt.AlignTop)
        self.cancel_btn = QPushButton("取消任务")
        self.cancel_btn.setObjectName("Ghost")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_task)
        header.addWidget(self.cancel_btn, 0, Qt.AlignTop)
        self.log_btn = QPushButton("日志")
        self.log_btn.setObjectName("Ghost")
        self.log_btn.clicked.connect(self.show_log)
        header.addWidget(self.log_btn, 0, Qt.AlignTop)
        self.exp_btn = QPushButton("导出目录")
        self.exp_btn.setObjectName("Ghost")
        self.exp_btn.clicked.connect(self.export_all)
        header.addWidget(self.exp_btn, 0, Qt.AlignTop)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.tab_ccd(), "① CCD / PSF")
        self.tabs.addTab(self.tab_pd(), "② 相位差反演")
        self.tabs.addTab(self.tab_fea(), "③B FEA / 66 项")
        self.tabs.addTab(self.tab_zmx(), "③A ZMX")
        self.tabs.addTab(self.tab_ctrl(), "③C S / 控制")
        self.tabs.addTab(self.tab_demo(), "④ 演示")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self._src_label = QLabel("-")
        self.statusBar().addPermanentWidget(self._src_label)
        self._build_menu()
        self._settings = QSettings("TelescopeEngineering", "QtNative")
        geo = self._settings.value("geometry")
        if geo is not None:
            try:
                self.restoreGeometry(geo)
            except Exception:
                pass
        self._log("Qt GUI started")

    # ---------- helpers ----------
    def _build_menu(self):
        bar = self.menuBar()
        m_file = bar.addMenu("文件")
        act_exp = QAction("导出全部...", self)
        act_exp.triggered.connect(lambda: self.export_all())
        m_file.addAction(act_exp)
        act_open = QAction("打开导出目录", self)
        act_open.triggered.connect(self.open_export_dir)
        m_file.addAction(act_open)
        m_file.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)
        m_view = bar.addMenu("视图")
        for idx, name in enumerate(["CCD / PSF", "相位差反演", "FEA / 66 项",
                                    "ZMX", "S / 控制", "演示"]):
            act = QAction("%d. %s" % (idx + 1, name), self)
            act.triggered.connect(lambda _=False, i=idx: self.tabs.setCurrentIndex(i))
            m_view.addAction(act)
        m_help = bar.addMenu("帮助")
        act_about = QAction("关于 / 标签说明", self)
        act_about.triggered.connect(self.show_about)
        m_help.addAction(act_about)

    def show_about(self):
        QMessageBox.information(
            self, "关于",
            "Telescope Engineering · Qt native\n\n"
            "标签约定：\n"
            "CORE-VERIFIED / SYNTHETIC-VALIDATED / EXTERNAL-CHECK /\n"
            "FEA-PROVISIONAL / GUI-BETA / UNVERIFIED\n\n"
            "数据来源: data/zemax（Zemax ZOS-API 真实求解）、\n"
            "data/fea_cases（Abaqus 求解）")

    def status(self, text, color=GOOD):
        self._log(text)
        self.badge.setText(text.upper() if len(text) < 18 else text)
        self.badge.setStyleSheet(
            "background:#0b1117;color:%s;border:1px solid #2a3a4a;border-radius:11px;padding:3px 10px;font-weight:600;" % color)
        self.statusBar().showMessage(text)

    def show_log(self):
        d = QDialog(self)
        d.setWindowTitle("Qt GUI log")
        d.resize(900, 520)
        lay = QVBoxLayout(d)
        te = QTextEdit(); te.setReadOnly(True)
        try:
            te.setPlainText(self.log_path.read_text(encoding="utf-8")[-200000:])
        except Exception as e:
            te.setPlainText(str(e))
        lay.addWidget(te)
        row = QHBoxLayout()
        copy = QPushButton("复制全部"); copy.setObjectName("Ghost")
        copy.clicked.connect(lambda: (QApplication.clipboard().setText(te.toPlainText()),
                                      self.status("日志已复制")))
        op = QPushButton("打开日志文件"); op.setObjectName("Ghost")
        op.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_path))))
        row.addWidget(copy); row.addWidget(op); row.addStretch(1)
        lay.addLayout(row)
        d.exec()

    def _log(self, text):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("%s  %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text))
        except Exception:
            pass

    def busy(self, text):
        self.status("运行中: " + text, WARN)
        self.tabs.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def ready(self):
        self.tabs.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.active_worker = None

    def _task_root(self):
        """Writable dir for job/result JSON (portable + sandbox friendly)."""
        candidates = [Path(sys.executable).resolve().parent / "qt_tasks",
                      ROOT / "results" / "qt_tasks"]
        for c in candidates:
            try:
                c.mkdir(parents=True, exist_ok=True)
                (c / ".write_test").write_text("ok", encoding="utf-8")
                (c / ".write_test").unlink()
                return c
            except Exception:
                continue
        raise RuntimeError("no writable qt_tasks directory")

    def _run_process(self, task, payload, done, label):
        if self.runner.busy():
            # was a silent return; tell the user why the click did nothing
            self.status("任务进行中，已忽略: " + label, WARN)
            return
        self._task_done = done
        self._task_label = label
        self.busy(label)
        try:
            self.runner.start(task, payload)
        except Exception as e:
            self.ready()
            self.status("失败: " + label, BAD)
            QMessageBox.critical(self, label, str(e))

    def _on_task_done(self, data):
        done = self._task_done
        label = self._task_label
        self._task_done = None
        self.ready()
        self._log("TASK ok %s %.2fs" % (label, self.runner.last_elapsed))
        self.status("完成: " + label)
        if done:
            done(data or {})

    def _on_task_failed(self, msg):
        label = self._task_label or "任务"
        self._task_done = None
        self.ready()
        self.status("失败: " + label, BAD)
        self._log("TASK fail %s %.2fs" % (label, self.runner.last_elapsed))
        self._log("ERROR " + label + "\n" + msg)
        box = QMessageBox(QMessageBox.Critical, label, str(msg)[:500], QMessageBox.Ok, self)
        box.setDetailedText(msg)
        box.exec()

    def _on_task_cancelled(self):
        label = self._task_label or "任务"
        self._task_done = None
        self.ready()
        self._log("TASK cancel %s %.2fs" % (label, self.runner.last_elapsed))
        self.status("已取消: " + label, WARN)

    def _shutdown_workers(self):
        """Detach and stop the in-process demo Worker QThreads.

        Worker.result/failed slots update tabs/logs; firing after closeEvent
        would run against already-deleted C++ widgets ("Internal C++ object
        already deleted"), and a QThread still running at interpreter exit can
        abort the process.  So: disconnect the slots first, give a running
        thread a short grace period (the demo render is sub-second), and as a
        last resort terminate() it before the event loop goes away.
        """
        workers = list(getattr(self, "_workers", []) or [])
        current = getattr(self, "active_worker", None)
        if current is not None and current not in workers:
            workers.append(current)
        self._workers = []
        self.active_worker = None
        for w in workers:
            for sig in (getattr(w, "result", None), getattr(w, "failed", None)):
                if sig is None:
                    continue
                try:
                    sig.disconnect()
                except Exception:
                    pass  # never connected / already disconnected
        for w in workers:
            try:
                if w.isRunning():
                    w.wait(2000)
                if w.isRunning():
                    w.terminate()
                    w.wait(1000)
            except Exception:
                pass  # thread already gone

    def closeEvent(self, event):
        try:
            self._settings.setValue("geometry", self.saveGeometry())
        except Exception:
            pass
        # cancel + join the demo QThread(s) first: their result/failed
        # callbacks must not fire against widgets that are about to die,
        # then stop the QProcess task exactly as before
        self._shutdown_workers()
        if self.runner.busy():
            self.runner.cancel()
        super().closeEvent(event)

    def cancel_task(self):
        if not self.runner.busy():
            if getattr(self, "active_worker", None) is not None:
                self.status("演示任务不可中断（短任务，请等待完成）", WARN)
            return
        if QMessageBox.question(self, "取消任务",
                                "终止当前子进程？服务计算会立即停止。") != QMessageBox.Yes:
            return
        self.runner.cancel()

    def _run(self, fn, done, label):
        self.busy(label)
        w = Worker(fn)
        self._workers.append(w)
        self.active_worker = w

        def finish(r):
            self.ready(); self.status("完成: " + label)
            done(r)

        def fail(e):
            self.ready(); self.status("失败: " + label, BAD)
            self._log("ERROR " + label + "\n" + e)
            box = QMessageBox(QMessageBox.Critical, label, e.split("\n")[0], QMessageBox.Ok, self)
            box.setDetailedText(e)
            box.exec()

        w.result.connect(finish); w.failed.connect(fail)
        w.finished.connect(lambda: self._workers.remove(w) if w in self._workers else None)
        w.start()
