@echo off
cd /d "%~dp0"
set "PATH=C:\Program Files\nodejs;%PATH%"
where npm >nul 2>nul || (echo. & echo ERROR: Node/npm not found. & pause & exit /b 1)
if not exist node_modules call npm install
echo.
echo Starting demo - a browser tab will open shortly.
echo Close this window to stop the server.
echo.
call npm run dev
echo.
echo Server stopped.
pause
