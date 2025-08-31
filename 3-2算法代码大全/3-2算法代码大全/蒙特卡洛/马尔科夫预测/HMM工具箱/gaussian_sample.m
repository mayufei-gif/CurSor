% 文件: gaussian_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function x = gsamp(mu, covar, nsamp)  % 详解: 执行语句


d = size(covar, 1);  % 详解: 赋值：将 size(...) 的结果保存到 d

mu = reshape(mu, 1, d);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu

[evec, eval] = eig(covar);  % 详解: 执行语句

coeffs = randn(nsamp, d)*sqrt(eval);  % 详解: 赋值：将 randn(...) 的结果保存到 coeffs

x = ones(nsamp, 1)*mu + coeffs*evec';  % 创建全 1 矩阵/数组  % 详解: 赋值：将 ones(...) 的结果保存到 x  % 详解: 赋值：将 ones(...) 的结果保存到 x




