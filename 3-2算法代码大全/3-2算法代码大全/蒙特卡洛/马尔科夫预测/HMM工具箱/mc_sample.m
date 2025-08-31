% 文件: mc_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function S = mc_sample(prior, trans, len, numex)  % 详解: 执行语句

if nargin==3  % 详解: 条件判断：if (nargin==3)
  numex = 1;  % 详解: 赋值：计算表达式并保存到 numex
end  % 详解: 执行语句

S = zeros(numex,len);  % 详解: 赋值：将 zeros(...) 的结果保存到 S
for i=1:numex  % 详解: for 循环：迭代变量 i 遍历 1:numex
  S(i, 1) = sample_discrete(prior);  % 详解: 调用函数：S(i, 1) = sample_discrete(prior)
  for t=2:len  % 详解: for 循环：迭代变量 t 遍历 2:len
    S(i, t) = sample_discrete(trans(S(i,t-1),:));  % 详解: 调用函数：S(i, t) = sample_discrete(trans(S(i,t-1),:))
  end  % 详解: 执行语句
end  % 详解: 执行语句




