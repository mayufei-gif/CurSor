% 文件: sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function x = sample(p, n)  % 详解: 执行语句

if nargin < 2  % 详解: 条件判断：if (nargin < 2)
  n = 1;  % 详解: 赋值：计算表达式并保存到 n
end  % 详解: 执行语句

cdf = cumsum(p(:));  % 详解: 赋值：将 cumsum(...) 的结果保存到 cdf
for i = 1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
  x(i) = sum(cdf < rand) + 1;  % 详解: 统计：求和/均值/中位数
end  % 详解: 执行语句




