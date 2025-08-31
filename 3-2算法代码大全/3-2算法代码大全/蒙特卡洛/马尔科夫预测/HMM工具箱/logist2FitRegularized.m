% 文件: logist2FitRegularized.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [net, niter] = logist2FitRegularized(labels, features, maxIter)  % 详解: 函数定义：logist2FitRegularized(labels, features, maxIter), 返回：net, niter

if nargin < 3, maxIter = 100; end  % 详解: 条件判断：if (nargin < 3, maxIter = 100; end)

[D  N] = size(features);  % 详解: 获取向量/矩阵尺寸
weightPrior = 0.5;  % 详解: 赋值：计算表达式并保存到 weightPrior
net = glm(D, 1, 'logistic', weightPrior);  % 详解: 赋值：将 glm(...) 的结果保存到 net
options = foptions;  % 详解: 赋值：计算表达式并保存到 options
options(14) = maxIter;  % 详解: 执行语句
[net, options] = glmtrain(net, options, features', labels(:));  % 执行语句  % 详解: 执行语句  % 详解: 执行语句
niter = options(14);  % 详解: 赋值：将 options(...) 的结果保存到 niter





