% 文件: setdiag.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function M = setdiag(M, v)  % 详解: 执行语句

n = length(M);  % 详解: 赋值：将 length(...) 的结果保存到 n
if length(v)==1  % 详解: 条件判断：if (length(v)==1)
  v = repmat(v, 1, n);  % 详解: 赋值：将 repmat(...) 的结果保存到 v
end  % 详解: 执行语句



J = 1:n+1:n^2;  % 详解: 赋值：计算表达式并保存到 J
M(J) = v;  % 详解: 执行语句






