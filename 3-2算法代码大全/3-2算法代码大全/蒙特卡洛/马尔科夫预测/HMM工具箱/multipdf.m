% 文件: multipdf.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = multipdf(x,theta)  % 详解: 执行语句

error(nargchk(2,2,nargin));  % 详解: 调用函数：error(nargchk(2,2,nargin))

if ndims(theta) > 2 | all(size(theta) > 1)  % 详解: 条件判断：if (ndims(theta) > 2 | all(size(theta) > 1))
    error('theta must be a vector');  % 详解: 调用函数：error('theta must be a vector')
end  % 详解: 执行语句

if ndims(x) > 2 | any(size(x) ~= size(theta))  % 详解: 条件判断：if (ndims(x) > 2 | any(size(x) ~= size(theta)))
    error('columns of X must have same length as theta');  % 详解: 调用函数：error('columns of X must have same length as theta')
end  % 详解: 执行语句


p = prod(theta .^ x);  % 详解: 赋值：将 prod(...) 的结果保存到 p
p = p .* factorial(sum(x)) ./ prod(factorial_v(x));  % 详解: 赋值：计算表达式并保存到 p


function r = factorial_v(x)  % 详解: 执行语句

if size(x,2) == 1  % 详解: 条件判断：if (size(x,2) == 1)
    x = x';  % 赋值：设置变量 x  % 详解: 赋值：计算表达式并保存到 x  % 详解: 赋值：计算表达式并保存到 x
end  % 详解: 执行语句

r = [];  % 详解: 赋值：计算表达式并保存到 r
for y = x  % 详解: for 循环：迭代变量 y 遍历 x
    r = [r factorial(y)];  % 详解: 赋值：计算表达式并保存到 r
end  % 详解: 执行语句



