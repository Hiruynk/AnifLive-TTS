@echo off
setlocal
cd /d "%~dp0"

set "ANIFLIVE_TTS_INFERENCE_LAUNCHER=%CD%\scripts\run_v14_inference.ps1"
if not exist "%ANIFLIVE_TTS_INFERENCE_LAUNCHER%" (
  echo [AnifLive-TTS] ERROR: Missing Docker inference launcher:
  echo   %ANIFLIVE_TTS_INFERENCE_LAUNCHER%
  pause
  exit /b 1
)

echo [AnifLive-TTS] Starting the v1.4 Linux Docker inference service in the foreground.
echo [AnifLive-TTS] Close with Ctrl+C; this terminal owns the inference container.
powershell -NoProfile -Command ^
  "$listener = Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($listener) { Write-Output $listener.OwningProcess; exit 1 }; exit 0" > "%TEMP%\aniflive-tts-port-9880.pid"
if errorlevel 1 (
  for /f "usebackq delims=" %%P in ("%TEMP%\aniflive-tts-port-9880.pid") do echo [AnifLive-TTS] Port 9880 is already in use by PID %%P. No process was stopped.
  del /q "%TEMP%\aniflive-tts-port-9880.pid" >nul 2>&1
  pause
  exit /b 1
)
del /q "%TEMP%\aniflive-tts-port-9880.pid" >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -File "%ANIFLIVE_TTS_INFERENCE_LAUNCHER%"
set "ANIFLIVE_TTS_API_EXIT_CODE=%ERRORLEVEL%"

echo.
echo [AnifLive-TTS] Docker inference stopped with exit code %ANIFLIVE_TTS_API_EXIT_CODE%.
pause
exit /b %ANIFLIVE_TTS_API_EXIT_CODE%
