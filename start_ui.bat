@echo off
chcp 65001 >nul
echo ========================================
echo   RL Training Framework - Web UI
echo ========================================
echo.

rem Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Error: streamlit is not installed!
    echo Install with: pip install streamlit
    echo.
    pause
    exit /b 1
)

rem Get local IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4" ^| findstr /v "127.0.0.1"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        set LOCAL_IP=%%b
        goto :found
    )
)
:found

echo Starting Streamlit server...
echo.
echo Access URLs:
echo   - Local:   http://localhost:8501
echo   - Network: http://%LOCAL_IP%:8501
echo.
echo Note: External devices on the same network can access this server.
echo       Make sure your firewall allows incoming connections on port 8501.
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
