% 文件: entropy.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function H = entropy(v, scale)  % 详解: 执行语句

if nargin < 2, scale = 0; end  % 详解: 条件判断：if (nargin < 2, scale = 0; end)

v = v + (v==0);  % 详解: 赋值：计算表达式并保存到 v
H = -1 * sum(v .* log2(v), 1);  % 详解: 赋值：计算表达式并保存到 H

if scale  % 详解: 条件判断：if (scale)
  n = size(v, 1);  % 详解: 赋值：将 size(...) 的结果保存到 n
  unif = normalise(ones(n,1));  % 详解: 赋值：将 normalise(...) 的结果保存到 unif
  H = H / entropy(unif);  % 详解: 赋值：计算表达式并保存到 H
end  % 详解: 执行语句




