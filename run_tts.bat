@echo off
setlocal
cd /d "%~dp0"
set "ANIFLIVE_TTS_PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":9880 .*LISTENING"') do set "ANIFLIVE_TTS_PORT_PID=%%P"
if defined ANIFLIVE_TTS_PORT_PID (
  echo ERROR: Port 9880 is already in use by PID %ANIFLIVE_TTS_PORT_PID%.
  echo Stop that process explicitly or choose another port. Nothing was terminated.
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Missing .venv. Run: py -3.11 -m venv .venv
  exit /b 1
)
".venv\Scripts\python.exe" -B local_tts_cf.py
exit /b %errorlevel%
