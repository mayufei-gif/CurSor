% 文件: standardize.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [S, mu, sigma2] = standardize(M, mu, sigma2)  % 详解: 函数定义：standardize(M, mu, sigma2), 返回：S, mu, sigma2

M = double(M);  % 详解: 赋值：将 double(...) 的结果保存到 M
if nargin < 2  % 详解: 条件判断：if (nargin < 2)
  mu = mean(M,2);  % 详解: 赋值：将 mean(...) 的结果保存到 mu
  sigma2 = std(M,0,2);  % 详解: 赋值：将 std(...) 的结果保存到 sigma2
  sigma2 = sigma2 + eps*(sigma2==0);  % 详解: 赋值：计算表达式并保存到 sigma2
end  % 详解: 执行语句

[nrows ncols] = size(M);  % 详解: 获取向量/矩阵尺寸
S = M - repmat(mu(:), [1 ncols]);  % 详解: 赋值：计算表达式并保存到 S
S = S ./ repmat(sigma2, [1 ncols]);  % 详解: 赋值：计算表达式并保存到 S





