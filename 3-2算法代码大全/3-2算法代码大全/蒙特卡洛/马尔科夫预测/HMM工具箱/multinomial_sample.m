% 文件: multinomial_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function Y = sample_cond_multinomial(X, M)  % 详解: 执行语句

if any(X(:)==0)  % 详解: 条件判断：if (any(X(:)==0))
  error('data must only contain positive integers')  % 详解: 调用函数：error('data must only contain positive integers')
end  % 详解: 执行语句

Y = zeros(size(X));  % 详解: 赋值：将 zeros(...) 的结果保存到 Y
for i=min(X(:)):max(X(:))  % 详解: for 循环：迭代变量 i 遍历 min(X(:)):max(X(:))
  ndx = find(X==i);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
  Y(ndx) = sample_discrete(M(i,:), length(ndx), 1);  % 详解: 调用函数：Y(ndx) = sample_discrete(M(i,:), length(ndx), 1)
end  % 详解: 执行语句






