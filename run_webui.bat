@echo off
setlocal
cd /d "%~dp0"

set "ANIFLIVE_TTS_WEBUI_URL=http://127.0.0.1:9890/"

if not exist ".venv\Scripts\aniflive-tts.exe" (
  echo [AnifLive-TTS] Missing .venv\Scripts\aniflive-tts.exe
  echo Install the project environment before starting the WebUI.
  pause
  exit /b 1
)

if not defined ANIFLIVE_TTS_WEBUI_UPSTREAM (
  for %%P in (9880 9882) do (
    powershell -NoProfile -Command ^
      "try { $health = Invoke-RestMethod -Uri 'http://127.0.0.1:%%P/health' -TimeoutSec 2; if ($health.ready) { exit 0 }; exit 2 } catch { exit 1 }"
    if not errorlevel 1 if not defined ANIFLIVE_TTS_WEBUI_UPSTREAM set "ANIFLIVE_TTS_WEBUI_UPSTREAM=http://127.0.0.1:%%P"
  )
)

if not defined ANIFLIVE_TTS_WEBUI_UPSTREAM (
  echo [AnifLive-TTS] No ready API was found at http://127.0.0.1:9880 or http://127.0.0.1:9882
  echo Start the AnifLive-TTS API first, then run this launcher again.
  pause
  exit /b 1
)

powershell -NoProfile -Command ^
  "try { $health = Invoke-RestMethod -Uri '%ANIFLIVE_TTS_WEBUI_UPSTREAM%/health' -TimeoutSec 5; if (-not $health.ready) { exit 2 } } catch { exit 1 }"
if errorlevel 1 (
  echo [AnifLive-TTS] The API is not ready at %ANIFLIVE_TTS_WEBUI_UPSTREAM%
  echo Start the AnifLive-TTS API first, then run this launcher again.
  pause
  exit /b 1
)

powershell -NoProfile -Command ^
  "if (Get-NetTCPConnection -LocalPort 9890 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
  echo [AnifLive-TTS] Port 9890 is already in use. No process was stopped.
  pause
  exit /b 1
)

echo [AnifLive-TTS] WebUI: %ANIFLIVE_TTS_WEBUI_URL%
echo [AnifLive-TTS] Close this window or press Ctrl+C to stop only the WebUI.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command ^
  "Start-Sleep -Seconds 2; Start-Process '%ANIFLIVE_TTS_WEBUI_URL%'"

call ".venv\Scripts\aniflive-tts.exe" webui ^
  --host 127.0.0.1 ^
  --port 9890 ^
  --upstream "%ANIFLIVE_TTS_WEBUI_UPSTREAM%"

echo.
echo [AnifLive-TTS] WebUI stopped with exit code %ERRORLEVEL%.
pause
