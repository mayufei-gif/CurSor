% 文件: sample_discrete.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function M = sample_discrete(prob, r, c)  % 详解: 执行语句

n = length(prob);  % 详解: 赋值：将 length(...) 的结果保存到 n

if nargin == 1  % 详解: 条件判断：if (nargin == 1)
  r = 1; c = 1;  % 详解: 赋值：计算表达式并保存到 r
elseif nargin == 2  % 详解: 条件判断：elseif (nargin == 2)
  c == r;  % 详解: 赋值：计算表达式并保存到 c
end  % 详解: 执行语句

R = rand(r, c);  % 详解: 赋值：将 rand(...) 的结果保存到 R
M = ones(r, c);  % 详解: 赋值：将 ones(...) 的结果保存到 M
cumprob = cumsum(prob(:));  % 详解: 赋值：将 cumsum(...) 的结果保存到 cumprob

if n < r*c  % 详解: 条件判断：if (n < r*c)
  for i = 1:n-1  % 详解: for 循环：迭代变量 i 遍历 1:n-1
    M = M + (R > cumprob(i));  % 详解: 赋值：计算表达式并保存到 M
  end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  cumprob2 = cumprob(1:end-1);  % 详解: 赋值：将 cumprob(...) 的结果保存到 cumprob2
  for i=1:r  % 详解: for 循环：迭代变量 i 遍历 1:r
    for j=1:c  % 详解: for 循环：迭代变量 j 遍历 1:c
      M(i,j) = sum(R(i,j) > cumprob2)+1;  % 详解: 统计：求和/均值/中位数
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句







