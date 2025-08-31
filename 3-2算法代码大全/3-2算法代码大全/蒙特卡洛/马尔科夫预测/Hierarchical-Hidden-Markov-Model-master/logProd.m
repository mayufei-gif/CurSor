% 文件: logProd.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function d = logProd(a, b)  % 详解: 执行语句
	if (isnan(a)) || (isnan(b))  % 详解: 条件判断：if ((isnan(a)) || (isnan(b)))
        d = NaN;  % 详解: 赋值：计算表达式并保存到 d
    else  % 详解: 条件判断：else 分支
        d = a+b;  % 详解: 赋值：计算表达式并保存到 d
    end  % 详解: 执行语句
    if a==-inf && b==-inf  % 详解: 条件判断：if (a==-inf && b==-inf)
        d = -inf;  % 详解: 赋值：计算表达式并保存到 d
    end  % 详解: 执行语句
end  % 详解: 执行语句



