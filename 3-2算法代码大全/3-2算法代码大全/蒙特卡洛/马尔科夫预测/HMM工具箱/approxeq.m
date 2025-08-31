% 文件: approxeq.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = approxeq(a, b, tol, rel)  % 详解: 执行语句

if nargin < 3, tol = 1e-2; end  % 详解: 条件判断：if (nargin < 3, tol = 1e-2; end)
if nargin < 4, rel = 0; end  % 详解: 条件判断：if (nargin < 4, rel = 0; end)

a = a(:);  % 详解: 赋值：将 a(...) 的结果保存到 a
b = b(:);  % 详解: 赋值：将 b(...) 的结果保存到 b
d = abs(a-b);  % 详解: 赋值：将 abs(...) 的结果保存到 d
if rel  % 详解: 条件判断：if (rel)
  p = ~any( (d ./ (abs(a)+eps)) > tol);  % 详解: 赋值：计算表达式并保存到 p
else  % 详解: 条件判断：else 分支
  p = ~any(d > tol);  % 详解: 赋值：计算表达式并保存到 p
end  % 详解: 执行语句





