@echo off
echo 正在关闭所有Chrome进程...
taskkill /f /im chrome.exe >nul 2>&1

echo 等待2秒...
timeout /t 2 /nobreak >nul

echo 启动Chrome调试模式...
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%~dp0temp_chrome_debug"

echo Chrome调试模式已启动，端口：9222
echo 请等待3秒让Chrome完全启动...
timeout /t 3 /nobreak >nul

echo 验证调试端口...
curl http://localhost:9222/json
pause