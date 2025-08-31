% 文件: pca_netlab.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [PCcoeff, PCvec] = pca(data, N)  % 详解: 函数定义：pca(data, N), 返回：PCcoeff, PCvec


if nargin == 1  % 详解: 条件判断：if (nargin == 1)
   N = size(data, 2);  % 详解: 赋值：将 size(...) 的结果保存到 N
end  % 详解: 执行语句

if nargout == 1  % 详解: 条件判断：if (nargout == 1)
   evals_only = logical(1);  % 详解: 赋值：将 logical(...) 的结果保存到 evals_only
else  % 详解: 条件判断：else 分支
   evals_only = logical(0);  % 详解: 赋值：将 logical(...) 的结果保存到 evals_only
end  % 详解: 执行语句

if N ~= round(N) | N < 1 | N > size(data, 2)  % 详解: 条件判断：if (N ~= round(N) | N < 1 | N > size(data, 2))
   error('Number of PCs must be integer, >0, < dim');  % 详解: 调用函数：error('Number of PCs must be integer, >0, < dim')
end  % 详解: 执行语句

if evals_only  % 详解: 条件判断：if (evals_only)
   PCcoeff = eigdec(cov(data), N);  % 详解: 赋值：将 eigdec(...) 的结果保存到 PCcoeff
else  % 详解: 条件判断：else 分支
  [PCcoeff, PCvec] = eigdec(cov(data), N);  % 详解: 执行语句
end  % 详解: 执行语句





