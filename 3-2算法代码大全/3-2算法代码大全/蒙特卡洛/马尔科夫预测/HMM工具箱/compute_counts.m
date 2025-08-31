% 文件: compute_counts.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function count = compute_counts(data, sz)  % 详解: 执行语句

assert(length(sz) == size(data, 1));  % 详解: 调用函数：assert(length(sz) == size(data, 1))
P = prod(sz);  % 详解: 赋值：将 prod(...) 的结果保存到 P
indices = subv2ind(sz, data'); % each row of data' is a case  % 详解: 赋值：将 subv2ind(...) 的结果保存到 indices
count = hist(indices, 1:P);  % 详解: 赋值：将 hist(...) 的结果保存到 count
count = myreshape(count, sz);  % 详解: 赋值：将 myreshape(...) 的结果保存到 count





