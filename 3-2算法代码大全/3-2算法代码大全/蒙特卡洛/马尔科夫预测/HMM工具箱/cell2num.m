% 文件: cell2num.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function N = cell2num(C)  % 详解: 执行语句




if isempty(C)  % 详解: 条件判断：if (isempty(C))
  N = [];  % 详解: 赋值：计算表达式并保存到 N
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

if any(cellfun('isempty', C))  % 详解: 条件判断：if (any(cellfun('isempty', C)))
  error('can''t convert cell array with empty cells to matrix')  % 详解: 调用函数：error('can''t convert cell array with empty cells to matrix')
end  % 详解: 执行语句

[nrows ncols] = size(C);  % 详解: 获取向量/矩阵尺寸
r = 0;  % 详解: 赋值：计算表达式并保存到 r
for i=1:nrows  % 详解: for 循环：迭代变量 i 遍历 1:nrows
  r = r + size(C{i,1}, 1);  % 详解: 赋值：计算表达式并保存到 r
end  % 详解: 执行语句
c = 0;  % 详解: 赋值：计算表达式并保存到 c
for j=1:ncols  % 详解: for 循环：迭代变量 j 遍历 1:ncols
  c = c + size(C{1,j}, 2);  % 详解: 赋值：计算表达式并保存到 c
end  % 详解: 执行语句
N = reshape(cat(1, C{:}), [r c]);  % 详解: 赋值：将 reshape(...) 的结果保存到 N





