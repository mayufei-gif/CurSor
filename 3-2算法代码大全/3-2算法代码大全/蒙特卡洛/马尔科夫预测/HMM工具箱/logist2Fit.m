% 文件: logist2Fit.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [beta, p] = logist2Fit(y, x, addOne, w)  % 详解: 函数定义：logist2Fit(y, x, addOne, w), 返回：beta, p

if nargin < 3, addOne = 1; end  % 详解: 条件判断：if (nargin < 3, addOne = 1; end)
if nargin < 4, w = 1; end  % 详解: 条件判断：if (nargin < 4, w = 1; end)

Ncases = size(x,2);  % 详解: 赋值：将 size(...) 的结果保存到 Ncases
if Ncases ~= length(y)  % 详解: 条件判断：if (Ncases ~= length(y))
  error(sprintf('size of data = %dx%d, size of labels=%d', size(x,1), size(x,2), length(y)))  % 详解: 调用函数：error(sprintf('size of data = %dx%d, size of labels=%d', size(x,1), size(x,2), length(y)))
end  % 详解: 执行语句
if addOne  % 详解: 条件判断：if (addOne)
  x = [x; ones(1,Ncases)];  % 详解: 赋值：计算表达式并保存到 x
end  % 详解: 执行语句
[beta, p] = logist2(y(:), x', w(:));  % 执行语句  % 详解: 执行语句  % 详解: 执行语句
beta = beta(:);  % 详解: 赋值：将 beta(...) 的结果保存到 beta




