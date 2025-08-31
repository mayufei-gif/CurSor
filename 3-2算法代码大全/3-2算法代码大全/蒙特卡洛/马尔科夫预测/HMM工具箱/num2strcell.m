% 文件: num2strcell.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function c = num2strcell(n, format)  % 详解: 执行语句

if nargin < 2, format = '%d'; end  % 详解: 条件判断：if (nargin < 2, format = '%d'; end)

N = length(n);  % 详解: 赋值：将 length(...) 的结果保存到 N
c = cell(1,N);  % 详解: 赋值：将 cell(...) 的结果保存到 c
for i=1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
  c{i} = sprintf(format, n(i));  % 详解: 执行语句
end  % 详解: 执行语句
  
  




