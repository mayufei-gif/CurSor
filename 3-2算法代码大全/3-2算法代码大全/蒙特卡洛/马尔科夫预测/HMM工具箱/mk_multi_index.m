% 文件: mk_multi_index.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function index = mk_multi_index(n, dims, vals)  % 详解: 执行语句

if n==0  % 详解: 条件判断：if (n==0)
  index = { 1 };  % 详解: 赋值：计算表达式并保存到 index
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

index = cell(1,n);  % 详解: 赋值：将 cell(...) 的结果保存到 index
for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
  index{i} = ':';  % 详解: 执行语句
end  % 详解: 执行语句
for i=1:length(dims)  % 详解: for 循环：迭代变量 i 遍历 1:length(dims)
  index{dims(i)} = vals(i);  % 详解: 执行语句
end  % 详解: 执行语句





