% 文件: marginalize_gaussian.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [muX, SXX] = marginalize_gaussian(mu, Sigma, X, Y, ns)  % 详解: 函数定义：marginalize_gaussian(mu, Sigma, X, Y, ns), 返回：muX, SXX

[muX, muY, SXX, SXY, SYX, SYY] = partition_matrix_vec(mu, Sigma, X, Y, ns);  % 详解: 执行语句






