# C# 开发环境配置指南

## 问题解决

✅ **已解决的问题：**
- `scriptcs` 命令无法识别错误
- VS Code 默认使用错误的 C# 运行器
- 文件锁定和进程冲突问题

## 新的开发流程

### 1. 创建新的 C# 项目

**方法一：使用 PowerShell 脚本（推荐）**
```powershell
# 在项目根目录运行
.\create_csharp_project.ps1 -ProjectName "你的项目名"
```

**方法二：手动创建**
```bash
# 进入 C# 目录
cd "C#"

# 创建新项目
dotnet new console -o 项目名 --force

# 进入项目目录
cd 项目名
```

### 2. 运行 C# 程序

**在 VS Code 中：**
- 打开项目文件夹（包含 .csproj 文件的目录）
- 按 `Ctrl + F5` 运行
- 或者按 `F5` 调试运行
- 或者右键选择 "Run Code"

**在终端中：**
```bash
# 确保在包含 .csproj 文件的目录中
dotnet run
```

### 3. 项目结构

```
C#/
├── RectangleCalc/          # 示例项目
│   ├── Program.cs          # 主程序文件
│   ├── RectangleCalc.csproj # 项目配置文件
│   ├── bin/                # 编译输出
│   └── obj/                # 临时文件
└── 你的新项目/
    ├── Program.cs
    ├── 项目名.csproj
    ├── bin/
    └── obj/
```

## 重要说明

### ❌ 不要这样做：
- 直接运行单个 `.cs` 文件
- 使用 `scriptcs` 命令
- 在没有 `.csproj` 文件的目录中运行

### ✅ 正确做法：
- 始终在包含 `.csproj` 文件的项目目录中工作
- 使用 `dotnet run` 命令
- 通过 VS Code 的内置运行功能

## VS Code 配置说明

已配置的功能：
- **Code Runner**: 自动检测项目并使用 `dotnet run`
- **OmniSharp**: 现代 .NET 支持
- **IntelliSense**: 智能代码补全
- **调试支持**: F5 调试，Ctrl+F5 运行
- **文件关联**: .cs 文件正确识别为 C#

## 故障排除

### 如果仍然出现 scriptcs 错误：
1. 重启 VS Code
2. 确保在正确的项目目录中
3. 检查是否存在 `.csproj` 文件
4. 使用 `dotnet --version` 确认 .NET SDK 已安装

### 如果程序无法运行：
1. 检查当前目录是否包含 `.csproj` 文件
2. 运行 `dotnet restore` 恢复依赖
3. 运行 `dotnet build` 检查编译错误
4. 确保 Program.cs 包含正确的 Main 方法或顶级语句

## 示例代码模板

```csharp
// Program.cs - 控制台应用模板
using System;

Console.WriteLine("Hello, World!");

// 或者使用传统 Main 方法
/*
using System;

namespace YourProject
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
        }
    }
}
*/
```

---

**配置完成！现在你可以愉快地进行 C# 开发了！** 🎉