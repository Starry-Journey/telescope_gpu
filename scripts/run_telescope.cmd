@echo off
REM Fast launch: use cmd/start to avoid the Explorer SmartScreen EXE lookup.
REM (Run Unblock-File once separately if the file was downloaded.)
setlocal
set EXE=%~dp0..\dist\TelescopeEngineeringQtNative\TelescopeEngineeringQtNative.exe
start "" "%EXE%"
endlocal
