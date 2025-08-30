# C# 零基础学习指南
## 以矩形面积计算器为例

### 📋 代码逐行详解

#### 📄 完整代码文件：`Program.cs`

```csharp
// Program.cs  (.NET 8, top-level statements) 
```
**第1行**：注释，说明这是程序文件，使用.NET 8的顶级语句特性
- `//` 表示单行注释
- 顶级语句：不需要Main方法，直接写代码即可

```csharp
using System; 
```
**第2行**：引入系统命名空间
- `using`：类似Python的import，引入其他命名空间
- `System`：包含基本类，如Console、Math等
- 这行让我们可以使用`Console.WriteLine()`等方法

---

### 🔧 自定义函数详解

```csharp
double ReadPositiveNumber(string name) 
```
**第4-5行**：定义函数
- `double`：返回值类型，双精度浮点数
- `ReadPositiveNumber`：函数名称，使用驼峰命名法
- `string name`：参数，字符串类型，表示要输入的数值名称

```csharp
{ 
    while (true) 
    {
```
**第6-8行**：无限循环
- `while (true)`：永远为真的条件，循环不会自动停止
- 需要手动使用`break`或`return`退出
- 用于反复要求用户输入直到输入正确

```csharp
        Console.Write($"{name}（正数）: "); 
```
**第9行**：输出提示信息
- `Console.Write()`：输出但不换行
- `$""`：字符串插值，可以插入变量
- `{name}`：会被参数的实际值替换

```csharp
        string? s = Console.ReadLine(); 
```
**第10行**：读取用户输入
- `string?`：可空字符串类型（C# 8.0新特性）
- `Console.ReadLine()`：读取一行用户输入，返回字符串
- `?`表示这个变量可以为null（空值）

```csharp
        if (string.IsNullOrWhiteSpace(s)) 
        { 
            Console.WriteLine("输入不能为空，请重试。"); 
            continue; 
        } 
```
**第11-16行**：空值检查
- `string.IsNullOrWhiteSpace(s)`：检查字符串是否为空或空白
- `Console.WriteLine()`：输出并换行
- `continue`：跳过本次循环剩余部分，重新开始循环

```csharp
        if (double.TryParse(s.Trim(), out double v)) 
        { 
            if (v > 0) return v; 
            Console.WriteLine("请输入大于 0 的数。"); 
        } 
        else 
        { 
            Console.WriteLine("无效数字，请输入例如 3.5 或 4。"); 
        } 
```
**第17-27行**：数值转换和验证
- `double.TryParse()`：尝试将字符串转换为数字
- `s.Trim()`：去除字符串两端的空格
- `out double v`：输出参数，转换成功时存储结果
- `v > 0`：检查是否为正数
- `return v`：返回有效的正数，退出函数

---

### 🎯 主程序逻辑

```csharp
try 
{ 
    Console.WriteLine("计算矩形面积"); 
```
**第31-33行**：异常处理开始
- `try`：开始异常处理块
- 如果try块内发生错误，会跳转到catch块

```csharp
    double width  = ReadPositiveNumber("宽度"); 
    double height = ReadPositiveNumber("高度"); 
```
**第34-35行**：调用自定义函数
- 两次调用`ReadPositiveNumber()`获取宽度和高度
- 确保输入的都是正数

```csharp
    double area = width * height; 
    Console.WriteLine($"矩形面积 = {area:N6}（宽度 × 高度）"); 
```
**第37-38行**：计算和输出结果
- `width * height`：计算矩形面积
- `$""`：字符串插值
- `{area:N6}`：数值格式化为6位小数

```csharp
} 
catch (Exception ex) 
{ 
    Console.Error.WriteLine($"发生错误: {ex.Message}"); 
    Environment.ExitCode = 1; 
}
```
**第40-44行**：异常处理
- `catch (Exception ex)`：捕获所有异常
- `Console.Error.WriteLine()`：向错误输出流写入
- `Environment.ExitCode = 1`：设置程序退出码为1（表示错误）

---

### 🏗️ 项目结构说明

#### 📁 项目文件：`juxingmianji.csproj`
```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>
</Project>
```

**关键配置解释：**
- `<OutputType>Exe</OutputType>`：生成可执行文件
- `<TargetFramework>net8.0</TargetFramework>`：使用.NET 8.0框架
- `<ImplicitUsings>enable</ImplicitUsings>`：启用隐式using指令
- `<Nullable>enable</Nullable>`：启用可空引用类型检查

---

### 🚀 运行方法

#### 方法1：命令行运行
```bash
cd "G:\E盘\工作项目文件\AI_Agent\Trae_Abroad\C#\juxingmianji"
dotnet run
```

#### 方法2：VS Code运行
1. 打开项目文件夹
2. 按 `Ctrl+F5` 运行
3. 或按 `F5` 调试运行

---

### 📚 学习建议

#### 🎯 初学者重点掌握
1. **变量声明**：`类型 变量名 = 值;`
2. **函数定义**：`返回值类型 函数名(参数列表) { ... }`
3. **条件判断**：`if (条件) { ... } else { ... }`
4. **循环**：`while (条件) { ... }`
5. **异常处理**：`try { ... } catch { ... }`

#### 🔍 继续学习方向
1. **数据类型**：int, string, bool, decimal等
2. **面向对象**：class, object, inheritance
3. **集合**：List<T>, Dictionary<TKey,TValue>
4. **LINQ**：查询语法
5. **异步编程**：async/await

#### 📖 推荐练习
1. 修改程序计算圆形面积
2. 添加三角形面积计算
3. 实现简单的计算器
4. 创建学生成绩管理系统

---

### ❓ 常见问题解答

**Q: 为什么使用`double`而不是`int`？**
A: `double`支持小数，适合计算面积；`int`只能表示整数。

**Q: `string?`的问号是什么意思？**
A: 表示这个变量可以为null，是C#的可空引用类型特性。

**Q: `continue`和`break`有什么区别？**
A: `continue`跳过本次循环，`break`完全退出循环。

**Q: 如何调试程序？**
A: 在VS Code中设置断点（点击行号左侧），然后按F5启动调试。

---

**恭喜！你已经掌握了C#基础编程的核心概念！继续加油！** 🎉