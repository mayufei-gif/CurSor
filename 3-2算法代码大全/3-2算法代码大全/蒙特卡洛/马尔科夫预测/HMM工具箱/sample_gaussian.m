% 文件: sample_gaussian.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function M = sample_gaussian(mu, Sigma, N)  % 详解: 执行语句

if nargin==2  % 详解: 条件判断：if (nargin==2)
  N = 1;  % 详解: 赋值：计算表达式并保存到 N
end  % 详解: 执行语句


mu = mu(:);  % 详解: 赋值：将 mu(...) 的结果保存到 mu
n=length(mu);  % 详解: 赋值：将 length(...) 的结果保存到 n
[U,D,V] = svd(Sigma);  % 详解: 执行语句
M = randn(n,N);  % 详解: 赋值：将 randn(...) 的结果保存到 M
M = (U*sqrt(D))*M + mu*ones(1,N);  % 详解: 赋值：计算表达式并保存到 M
M = M';  % 赋值：设置变量 M  % 详解: 赋值：计算表达式并保存到 M  % 详解: 赋值：计算表达式并保存到 M





