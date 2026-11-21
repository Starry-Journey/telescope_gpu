# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — 从仓库根运行：py -3.13 -m PyInstaller packaging/windows/qt_app.spec
import os
from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, '..', '..'))
ICON = os.path.join(SPECPATH, 'icon.ico')

hiddenimports = collect_submodules('products') + collect_submodules('desktop') + ['nlopt']

a = Analysis(
    [os.path.join(ROOT, 'desktop', 'qt_app.py')],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, 'configs'), 'configs'),
        (os.path.join(ROOT, 'data/zemax'), 'data/zemax'),
        (os.path.join(ROOT, 'data/fea_cases'), 'data/fea_cases'),
    ],
    hiddenimports=hiddenimports,
    excludes=['torch', 'torchvision', 'torchaudio', 'ml', 'IPython', 'jupyter',
              'pytest', 'PySide6.WebEngine', 'PySide6.Pdf'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name='TelescopeEngineeringQtNative', debug=False,
          strip=False, upx=False, console=False, icon=ICON)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name='TelescopeEngineeringQtNative')
