@echo off
rem Windows 桌面端打包：PyInstaller onedir + Inno Setup 向导安装包
rem 前置：py -3.13 环境含 PySide6/numpy/scipy/astropy/matplotlib/nlopt/pyinstaller；Inno Setup 6（ISCC）
rem 中文向导语言包：将 ChineseSimplified.isl 放在本目录（github.com/jrsoftware/issrc Files/Languages）
cd /d %~dp0..\..
py -3.13 -m PyInstaller --noconfirm packaging\windows\qt_app.spec || goto :err
copy /y packaging\windows\README_安装说明.txt dist\TelescopeEngineeringQtNative\ >nul
if not exist ChineseSimplified.isl copy packaging\windows\ChineseSimplified.isl . >nul 2>&1
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows\installer.iss || goto :err
echo DONE: dist\TelescopeEngineeringQtNative + packaging\windows\installer_out
exit /b 0
:err
echo BUILD FAILED
exit /b 1
