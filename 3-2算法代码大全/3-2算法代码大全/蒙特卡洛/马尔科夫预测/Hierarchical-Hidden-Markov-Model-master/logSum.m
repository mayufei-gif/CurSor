% 文件: logSum.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function d = logSum(a, b)  % 详解: 执行语句
    if (isnan(a)) || (isnan(b))  % 详解: 条件判断：if ((isnan(a)) || (isnan(b)))
        if (isnan(a))  % 详解: 条件判断：if ((isnan(a)))
            d = b;  % 详解: 赋值：计算表达式并保存到 d
        else  % 详解: 条件判断：else 分支
            d = a;  % 详解: 赋值：计算表达式并保存到 d
        end  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
        if a>b  % 详解: 条件判断：if (a>b)
            d = a + log(1 + exp(b-a));  % 详解: 赋值：计算表达式并保存到 d
        else  % 详解: 条件判断：else 分支
            d = b + log(1 + exp(a-b));  % 详解: 赋值：计算表达式并保存到 d
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    if a==-inf && b==-inf  % 详解: 条件判断：if (a==-inf && b==-inf)
        d = -inf;  % 详解: 赋值：计算表达式并保存到 d
    end  % 详解: 执行语句
end  % 详解: 执行语句



