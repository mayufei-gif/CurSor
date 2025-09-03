# Python代码编写教程完整备份脚本
# 创建时间: $(Get-Date)

# 设置源目录和备份目录
$sourceDir = "g:\E盘\工作项目文件\AI_Agent\Trae_Abroad\Python代码编写教程"
$backupDir = "g:\E盘\工作项目文件\AI_Agent\Trae_Abroad\Python备份_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

Write-Host "开始创建Python代码编写教程的完整备份..." -ForegroundColor Green
Write-Host "源目录: $sourceDir" -ForegroundColor Yellow
Write-Host "备份目录: $backupDir" -ForegroundColor Yellow

# 创建备份目录
if (!(Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    Write-Host "✓ 备份目录已创建" -ForegroundColor Green
}

# 复制所有文件和目录
Write-Host "正在复制文件..." -ForegroundColor Cyan

# 使用robocopy进行高效复制
$robocopyOptions = @(
    "$sourceDir"
    "$backupDir"
    "/E"           # 复制所有子目录，包括空目录
    "/COPY:DAT"    # 复制数据、属性、时间戳
    "/R:3"         # 重试3次
    "/W:5"         # 等待5秒
    "/NFL"         # 不显示文件名
    "/NDL"         # 不显示目录名
    "/NP"          # 不显示进度
    "/NJH"         # 不显示作业头
    "/NJS"         # 不显示作业摘要
)

$process = Start-Process -FilePath "robocopy" -ArgumentList $robocopyOptions -Wait -PassThru -NoNewWindow

if ($process.ExitCode -le 7) {
    Write-Host "✓ 文件复制完成" -ForegroundColor Green
    
    # 创建压缩包
    Write-Host "正在创建压缩包..." -ForegroundColor Cyan
    $zipPath = "$backupDir.zip"
    
    if (Get-Command Compress-Archive -ErrorAction SilentlyContinue) {
        Compress-Archive -Path "$backupDir\*" -DestinationPath $zipPath -Force
        Write-Host "✓ 压缩包已创建: $zipPath" -ForegroundColor Green
    } else {
        Write-Host "⚠ 无法创建压缩包，请手动压缩目录: $backupDir" -ForegroundColor Yellow
    }
    
    # 显示备份统计
    $fileCount = (Get-ChildItem -Path $backupDir -Recurse -File).Count
    $dirCount = (Get-ChildItem -Path $backupDir -Recurse -Directory).Count
    $totalSize = (Get-ChildItem -Path $backupDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
    
    Write-Host "`n备份完成统计:" -ForegroundColor Green
    Write-Host "文件数量: $fileCount" -ForegroundColor White
    Write-Host "目录数量: $dirCount" -ForegroundColor White
    Write-Host "总大小: $([math]::Round($totalSize, 2)) MB" -ForegroundColor White
    Write-Host "备份位置: $backupDir" -ForegroundColor White
    
} else {
    Write-Host "❌ 文件复制失败，错误代码: $($process.ExitCode)" -ForegroundColor Red
}

# 可选：打开备份目录
$choice = Read-Host "是否打开备份目录? (Y/N)"
if ($choice -eq "Y" -or $choice -eq "y") {
    Start-Process explorer.exe -ArgumentList $backupDir
}

Write-Host "`n备份脚本执行完毕!" -ForegroundColor Green
Write-Host "你可以随时运行此脚本创建新的备份" -ForegroundColor Cyan