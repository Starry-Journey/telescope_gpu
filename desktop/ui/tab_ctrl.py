"""Tab ③C S / 控制: sensitivity matrix, control solve, measured-calibration demo.

Split from desktop/gui.py (old lines 1113-1194) into the TabCtrl mixin.
Shared ``self`` attributes: last_s comes from AppShell.__init__;
ctrl_img/ctrl_log are created by this tab's ``tab_ctrl`` builder.
Cross-mixin helpers: _run_process (AppShell).  matplotlib stays lazily
imported inside _show_matrix_heatmap with the Agg backend, as before.
"""
from __future__ import annotations

import io

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget


class TabCtrl:
    """Methods for the ③C S / 控制 tab (no own __init__)."""

    def tab_ctrl(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        s_btn = QPushButton("计算 M2 灵敏度 S（Zemax 优先）")
        s_btn.clicked.connect(self.run_s)
        u_btn = QPushButton("求解控制量 u"); u_btn.clicked.connect(self.run_u)
        cal_btn = QPushButton("实测标定演示（CCD+PD 合成闭环）")
        cal_btn.clicked.connect(self.run_sens_demo)
        row.addWidget(s_btn); row.addWidget(u_btn); row.addWidget(cal_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.ctrl_img = QLabel("S heatmap"); self.ctrl_img.setAlignment(Qt.AlignCenter)
        self.ctrl_img.setMinimumSize(560, 260)
        self.ctrl_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        lay.addWidget(self.ctrl_img, 2)
        self.ctrl_log = QTextEdit(); self.ctrl_log.setReadOnly(True); self.ctrl_log.setMaximumHeight(170)
        lay.addWidget(self.ctrl_log, 1)
        return w

    def _show_matrix_heatmap(self, S, row_labels, col_labels, title, target):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            S = np.asarray(S, dtype=float)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            im = ax.imshow(S, cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(S.shape[1]))
            ax.set_xticklabels(col_labels, rotation=25, ha="right")
            ax.set_yticks(range(S.shape[0])); ax.set_yticklabels(row_labels)
            for i in range(S.shape[0]):
                for j in range(S.shape[1]):
                    ax.text(j, i, "%.2f" % S[i, j], ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            pix = QPixmap(); pix.loadFromData(buf.getvalue())
            target.setPixmap(pix.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.ctrl_log.append("heatmap error: " + str(e))

    def run_s(self):
        self._run_process("s", {}, self._on_s, "计算 S")

    def _on_s(self, d):
        self.last_s = d
        self.ctrl_log.setPlainText(
            "source=%s label=%s\nS=%s\nrank=%s condition=%s nullspace=%s\n%s" % (
                d.get("source"), d.get("label"), d.get("values"),
                d.get("rank"), d.get("condition"), d.get("nullspace"),
                d.get("note", "")))
        self._show_matrix_heatmap(d["values"], d.get("target_names", []),
                                  d.get("control_names", []),
                                  "Sensitivity S [waves/(mm or mrad)] — %s" %
                                  (d.get("source", "")), self.ctrl_img)

    def run_sens_demo(self):
        self._run_process("sens_demo", {}, self._on_sens_demo, "实测标定")

    def _on_sens_demo(self, d):
        S = d.get("S") or []
        self.ctrl_log.setPlainText(
            "实测标定（合成闭环） label=%s\n"
            "S_meas err_rms=%.4f corr=%.4f rank=%s cond=%.2f\n"
            "hysteresis=%s\nper-pose n=%s drift_rms=%.2e\n%s" % (
                d.get("label"), d.get("err_rms", float("nan")),
                d.get("corr", float("nan")), d.get("rank"),
                d.get("condition", float("nan")),
                d.get("hysteresis"), d.get("n_poses"),
                d.get("drift_rms", float("nan")), d.get("note", "")))
        self._show_matrix_heatmap(S, d.get("terms", []),
                                  d.get("dofs", d.get("dof_names", [])),
                                  "Measured S (synthetic closed loop) [waves/(mm or mrad)]",
                                  self.ctrl_img)

    def run_u(self):
        self._run_process("u", {"target": [0.23377, -0.09266, 0.04633]},
                          self._on_u, "求解 u")

    def _on_u(self, d):
        self.ctrl_log.setPlainText("u=%s\nresidual=%s" % (d.get("control"), d.get("residual")))
