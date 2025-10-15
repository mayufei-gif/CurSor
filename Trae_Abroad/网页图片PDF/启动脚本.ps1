# 网页图书PDF化工具 - 一键启动脚本
# 使用方法：在PowerShell中执行 .\启动脚本.ps1

Write-Host "🚀 网页图书PDF化工具启动中..." -ForegroundColor Green

# 检查当前目录
$currentDir = Get-Location
Write-Host "📁 当前目录: $currentDir" -ForegroundColor Yellow

# 检查必要文件
if (-not (Test-Path "cli.py")) {
    Write-Host "❌ 错误：找不到cli.py文件，请确保在正确的项目目录中运行此脚本" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

# 检查Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误：未找到Python，请先安装Python 3.8+" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

# 步骤1：关闭现有Chrome进程
Write-Host "`n🔄 步骤1：关闭现有Chrome进程..." -ForegroundColor Cyan
Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# 步骤2：启动Chrome调试模式
Write-Host "🔄 步骤2：启动Chrome调试模式..." -ForegroundColor Cyan

# 尝试不同的Chrome路径
$chromePaths = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)

$chromeFound = $false
foreach ($chromePath in $chromePaths) {
    if (Test-Path $chromePath) {
        Write-Host "✅ 找到Chrome: $chromePath" -ForegroundColor Green
        
        # 创建用户数据目录
        $userDataDir = Join-Path $currentDir "temp_chrome_debug"
        if (-not (Test-Path $userDataDir)) {
            New-Item -ItemType Directory -Path $userDataDir -Force | Out-Null
        }
        
        # 启动Chrome
        Start-Process -FilePath $chromePath -ArgumentList "--remote-debugging-port=9222", "--user-data-dir=`"$userDataDir`""
        $chromeFound = $true
        break
    }
}

if (-not $chromeFound) {
    Write-Host "❌ 错误：未找到Chrome浏览器，请先安装Google Chrome" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

# 步骤3：等待Chrome启动
Write-Host "🔄 步骤3：等待Chrome启动..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# 步骤4：验证调试端口
Write-Host "🔄 步骤4：验证调试端口..." -ForegroundColor Cyan
$maxRetries = 10
$retryCount = 0

do {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9222/json/version" -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Chrome调试端口连接成功" -ForegroundColor Green
            break
        }
    } catch {
        $retryCount++
        if ($retryCount -lt $maxRetries) {
            Write-Host "⏳ 等待Chrome启动... ($retryCount/$maxRetries)" -ForegroundColor Yellow
            Start-Sleep -Seconds 2
        } else {
            Write-Host "❌ 错误：无法连接到Chrome调试端口" -ForegroundColor Red
            Write-Host "请手动检查Chrome是否正常启动" -ForegroundColor Yellow
            Read-Host "按回车键退出"
            exit 1
        }
    }
} while ($retryCount -lt $maxRetries)

# 步骤5：启动浏览器控制程序
Write-Host "`n🔄 步骤5：启动浏览器控制程序..." -ForegroundColor Cyan
Write-Host "📋 程序启动后，请按照提示操作：" -ForegroundColor Yellow
Write-Host "   1. 在Chrome中打开图书网站并登录" -ForegroundColor White
Write-Host "   2. 调整页面缩放到67%" -ForegroundColor White
Write-Host "   3. 回到终端选择截图模式" -ForegroundColor White
Write-Host "   4. 推荐选择'2'进行自动化批量截图" -ForegroundColor White

Write-Host "`n🚀 正在启动浏览器控制程序..." -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray

# 启动主程序
python cli.py browser --connect-existing --debug-port 9222 --start-url blank --viewport-w 1600 --viewport-h 1200

Write-Host "`n🎉 程序已退出，感谢使用！" -ForegroundColor Green
Read-Host "按回车键关闭"