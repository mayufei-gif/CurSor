% 文件: est_transmat.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [A,C] = est_transmat(seq)  % 详解: 函数定义：est_transmat(seq), 返回：A,C


C = full(sparse(seq(1:end-1), seq(2:end),1));  % 详解: 赋值：将 full(...) 的结果保存到 C
A = mk_stochastic(C);  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 A




