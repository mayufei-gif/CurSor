% 文件: transmat_train_observed.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [transmat, initState] = transmat_train_observed(labels,  nstates, varargin)  % 详解: 函数定义：transmat_train_observed(labels, nstates, varargin), 返回：transmat, initState

[dirichletPriorWeight, mkSymmetric, other] = process_options(...  % 详解: 执行语句
    varargin, 'dirichletPriorWeight', 0, 'mkSymmetric', 0);  % 详解: 执行语句

if ~iscell(labels)  % 详解: 条件判断：if (~iscell(labels))
  [numex T] = size(labels);  % 详解: 获取向量/矩阵尺寸
  if T==1  % 详解: 条件判断：if (T==1)
    labels = labels';  % 赋值：设置变量 labels  % 详解: 赋值：计算表达式并保存到 labels  % 详解: 赋值：计算表达式并保存到 labels
  end  % 详解: 执行语句
  labels = num2cell(labels,2);  % 详解: 赋值：将 num2cell(...) 的结果保存到 labels
end  % 详解: 执行语句
numex = length(labels);  % 详解: 赋值：将 length(...) 的结果保存到 numex

counts = zeros(nstates, nstates);  % 详解: 赋值：将 zeros(...) 的结果保存到 counts
counts1 = zeros(nstates,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 counts1
for s=1:numex  % 详解: for 循环：迭代变量 s 遍历 1:numex
  labs = labels{s}; labs = labs(:)';  % 赋值：设置变量 labs  % 详解: 赋值：计算表达式并保存到 labs  % 详解: 赋值：计算表达式并保存到 labs
  dat = [labs(1:end-1); labs(2:end)];  % 详解: 赋值：计算表达式并保存到 dat
  counts = counts + compute_counts(dat, [nstates nstates]);  % 详解: 赋值：计算表达式并保存到 counts
  q = labs(1);  % 详解: 赋值：将 labs(...) 的结果保存到 q
  counts1(q) = counts1(q) + 1;  % 详解: 执行语句
end  % 详解: 执行语句
pseudo_counts = dirichletPriorWeight*ones(nstates, nstates);  % 详解: 赋值：计算表达式并保存到 pseudo_counts
if mkSymmetric  % 详解: 条件判断：if (mkSymmetric)
  counts = counts + counts';  % 赋值：设置变量 counts  % 详解: 赋值：计算表达式并保存到 counts  % 详解: 赋值：计算表达式并保存到 counts
end  % 详解: 执行语句
transmat = mk_stochastic(counts + pseudo_counts);  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat
initState = normalize(counts1 + dirichletPriorWeight*ones(nstates,1));  % 详解: 赋值：将 normalize(...) 的结果保存到 initState

  




