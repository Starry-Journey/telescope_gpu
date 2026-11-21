"""UI package for the native Qt desktop frontend (split from desktop/gui.py).

The public contract stays ``desktop.gui``: the facade re-exports QtApp,
ScaledImageLabel, Worker, run_gui and the plot/path/theme helpers, so every
historical import path keeps working.

Modules here are implementation details and may import each other plus
PySide6/stdlib — never ``desktop.gui`` (the facade imports this package).

Layout:
    theme.py          shared constants (DEFOCUS, QSS, GOOD/BAD/WARN)
    paths.py          ROOT, writable-dir/export-dir helpers
    image_widgets.py  ScaledImageLabel
    worker_thread.py  Worker (QThread for short in-process jobs)
    plot_helpers.py   _png/_fea_png/_bar_png/_frames_png (matplotlib lazy)
    app_shell.py      AppShell mixin: window shell, task glue, closeEvent
    export_actions.py ExportActions mixin: export/save actions
    tab_*.py          one mixin per tab (ccd / pd / fea / zmx / ctrl / demo)
"""
