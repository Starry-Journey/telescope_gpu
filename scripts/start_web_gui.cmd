@echo off
REM Start the verified Web GUI on port 8000 if it is not already running.
REM The desktop EXE launcher uses 8765; this script is the reliable fallback.
setlocal
set DISTRO=Ubuntu-20.04
set REPO=/home/legend-star/telescope_gpu

wsl -d %DISTRO% -- bash -lc "curl -s --max-time 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1 || (cd %REPO% && nohup python3 -m uvicorn web.api.main:app --host 0.0.0.0 --port 8000 >/tmp/ccd_web_desktop.log 2>&1 &)"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000/"
endlocal
