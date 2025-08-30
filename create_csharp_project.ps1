# PowerShell脚本：创建C#控制台项目
param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectName
)

# 设置项目路径
$ProjectPath = "$PSScriptRoot\C#\$ProjectName"

# 检查项目是否已存在
if (Test-Path $ProjectPath) {
    Write-Host "项目 '$ProjectName' 已存在，正在删除..." -ForegroundColor Yellow
    Remove-Item -Path $ProjectPath -Recurse -Force
}

# 创建新的控制台项目
Write-Host "正在创建项目 '$ProjectName'..." -ForegroundColor Green
dotnet new console -o $ProjectPath --force

# 切换到项目目录
Set-Location $ProjectPath

# 显示项目信息
Write-Host "\n项目创建成功！" -ForegroundColor Green
Write-Host "项目路径: $ProjectPath" -ForegroundColor Cyan
Write-Host "\n使用方法:" -ForegroundColor Yellow
Write-Host "1. cd '$ProjectPath'" -ForegroundColor White
Write-Host "2. dotnet run" -ForegroundColor White
Write-Host "\n或者在VS Code中按 Ctrl+F5 运行" -ForegroundColor White

# 返回原目录
Set-Location $PSScriptRoot