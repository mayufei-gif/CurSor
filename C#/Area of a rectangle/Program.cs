// Program.cs  (.NET 8, top-level statements) 
using System; 

double ReadPositiveNumber(string name) 
{ 
    while (true) 
    { 
        Console.Write($"{name}（正数）: "); 
        string? s = Console.ReadLine(); 
        if (string.IsNullOrWhiteSpace(s)) 
        { 
            Console.WriteLine("输入不能为空，请重试。"); 
            continue; 
        } 
        if (double.TryParse(s.Trim(), out double v)) 
        { 
            if (v > 0) return v; 
            Console.WriteLine("请输入大于 0 的数。"); 
        } 
        else 
        { 
            Console.WriteLine("无效数字，请输入例如 3.5 或 4。"); 
        } 
    } 
} 

try 
{ 
    Console.WriteLine("计算矩形面积"); 
    double width  = ReadPositiveNumber("宽度"); 
    double height = ReadPositiveNumber("高度"); 

    double area = width * height; 
    Console.WriteLine($"矩形面积 = {area:N6}（宽度 × 高度）"); 
} 
catch (Exception ex) 
{ 
    Console.Error.WriteLine($"发生错误: {ex.Message}"); 
    Environment.ExitCode = 1; 
}
