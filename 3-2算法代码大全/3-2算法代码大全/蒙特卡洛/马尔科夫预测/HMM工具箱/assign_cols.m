% 文件: assign_cols.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function M = assign_cols(cols, vals, M)  % 详解: 执行语句

if nargin < 3  % 详解: 条件判断：if (nargin < 3)
  nr = length(cols);  % 详解: 赋值：将 length(...) 的结果保存到 nr
  nc = max(cols);  % 详解: 赋值：将 max(...) 的结果保存到 nc
  M = zeros(nr, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 M
else  % 详解: 条件判断：else 分支
  [nr nc] = size(M);  % 详解: 获取向量/矩阵尺寸
end  % 详解: 执行语句

if 0  % 详解: 条件判断：if (0)
for r=1:nr  % 详解: for 循环：迭代变量 r 遍历 1:nr
  M(r, cols(r)) = vals(r);  % 详解: 调用函数：M(r, cols(r)) = vals(r)
end  % 详解: 执行语句
end  % 详解: 执行语句

if 1  % 详解: 条件判断：if (1)
rows = 1:nr;  % 详解: 赋值：计算表达式并保存到 rows
ndx = subv2ind([nr nc], [rows(:) cols(:)]);  % 详解: 赋值：将 subv2ind(...) 的结果保存到 ndx
M(ndx) = vals;  % 详解: 执行语句
end  % 详解: 执行语句




