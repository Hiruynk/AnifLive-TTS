@echo off
setlocal
cd /d "%~dp0"

set "ANIFLIVE_TTS_WEBUI_URL=http://127.0.0.1:9890/"
set "PYTHONPATH=%CD%\src"
set "ANIFLIVE_TTS_PYTHON_EXE="
set "ANIFLIVE_TTS_PYTHON_ARGS="

if defined ANIFLIVE_TTS_PYTHON (
  call :validate_python "%ANIFLIVE_TTS_PYTHON%" ""
  if errorlevel 1 (
    echo [AnifLive-TTS WebUI] ANIFLIVE_TTS_PYTHON cannot start this WebUI:
    echo   %ANIFLIVE_TTS_PYTHON%
    echo [AnifLive-TTS WebUI] Use Python 3.10-3.12 with fastapi, httpx, pydantic, and uvicorn available.
    echo [AnifLive-TTS WebUI] No packages were installed or changed.
    pause
    exit /b 1
  )
  set "ANIFLIVE_TTS_PYTHON_EXE=%ANIFLIVE_TTS_PYTHON%"
  goto python_ready
)

if exist ".venv\Scripts\python.exe" (
  call :validate_python "%CD%\.venv\Scripts\python.exe" ""
  if not errorlevel 1 (
    set "ANIFLIVE_TTS_PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    goto python_ready
  )
)

where python.exe >nul 2>&1
if not errorlevel 1 (
  call :validate_python "python.exe" ""
  if not errorlevel 1 (
    set "ANIFLIVE_TTS_PYTHON_EXE=python.exe"
    goto python_ready
  )
)

where py.exe >nul 2>&1
if not errorlevel 1 (
  for %%V in (-3.12 -3.11 -3.10) do (
    call :validate_python "py.exe" "%%V"
    if not errorlevel 1 (
      set "ANIFLIVE_TTS_PYTHON_EXE=py.exe"
      set "ANIFLIVE_TTS_PYTHON_ARGS=%%V"
      goto python_ready
    )
  )
)

echo [AnifLive-TTS WebUI] No usable WebUI Python environment was found.
echo [AnifLive-TTS WebUI] Checked the repository .venv, python.exe on PATH, and Python Launcher 3.12/3.11/3.10.
echo [AnifLive-TTS WebUI] A usable interpreter needs Python 3.10-3.12 with fastapi, httpx, pydantic, and uvicorn available.
echo [AnifLive-TTS WebUI] You can set ANIFLIVE_TTS_PYTHON to an existing compatible python.exe and run this launcher again.
echo [AnifLive-TTS WebUI] No packages were installed or changed.
pause
exit /b 1

:python_ready

if not defined ANIFLIVE_TTS_WEBUI_UPSTREAM (
  for %%P in (9880 9882) do (
    powershell -NoProfile -Command ^
      "try { $health = Invoke-RestMethod -Uri 'http://127.0.0.1:%%P/health' -TimeoutSec 2; if ($health.ready) { exit 0 }; exit 2 } catch { exit 1 }"
    if not errorlevel 1 if not defined ANIFLIVE_TTS_WEBUI_UPSTREAM set "ANIFLIVE_TTS_WEBUI_UPSTREAM=http://127.0.0.1:%%P"
  )
)

if not defined ANIFLIVE_TTS_WEBUI_UPSTREAM (
  set "ANIFLIVE_TTS_WEBUI_UPSTREAM=http://127.0.0.1:9880"
  echo [AnifLive-TTS WebUI] Inference API is offline. Synthesis will become available when the API starts.
)

powershell -NoProfile -Command ^
  "if (Get-NetTCPConnection -LocalPort 9890 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
  echo [AnifLive-TTS WebUI] Port 9890 is already in use. No process was stopped.
  pause
  exit /b 1
)

echo [AnifLive-TTS WebUI] WebUI: %ANIFLIVE_TTS_WEBUI_URL%
echo [AnifLive-TTS WebUI] Python: %ANIFLIVE_TTS_PYTHON_EXE% %ANIFLIVE_TTS_PYTHON_ARGS%
echo [AnifLive-TTS WebUI] Close this window or press Ctrl+C to stop the WebUI.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command ^
  "Start-Sleep -Seconds 2; Start-Process '%ANIFLIVE_TTS_WEBUI_URL%'"

call "%ANIFLIVE_TTS_PYTHON_EXE%" %ANIFLIVE_TTS_PYTHON_ARGS% -m aniflive_tts webui ^
  --host 127.0.0.1 ^
  --port 9890 ^
  --upstream "%ANIFLIVE_TTS_WEBUI_UPSTREAM%" ^
  --surface classic
set "ANIFLIVE_TTS_WEBUI_EXIT_CODE=%ERRORLEVEL%"

echo.
echo [AnifLive-TTS WebUI] WebUI stopped with exit code %ANIFLIVE_TTS_WEBUI_EXIT_CODE%.
pause
exit /b %ANIFLIVE_TTS_WEBUI_EXIT_CODE%

:validate_python
"%~1" %~2 -c "import sys; assert sys.version_info[:2] in ((3, 10), (3, 11), (3, 12)); import fastapi, httpx, pydantic, uvicorn; import aniflive_tts.webui" >nul 2>&1
exit /b %ERRORLEVEL%
