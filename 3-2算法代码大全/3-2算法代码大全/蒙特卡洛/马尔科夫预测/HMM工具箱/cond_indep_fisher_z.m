% 文件: cond_indep_fisher_z.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [CI, r, p] = cond_indep_fisher_z(X, Y, S, C, N, alpha)  % 详解: 函数定义：cond_indep_fisher_z(X, Y, S, C, N, alpha), 返回：CI, r, p

if nargin < 6, alpha = 0.05; end  % 详解: 条件判断：if (nargin < 6, alpha = 0.05; end)

r = partial_corr_coef(C, X, Y, S);  % 详解: 赋值：将 partial_corr_coef(...) 的结果保存到 r
z = 0.5*log( (1+r)/(1-r) );  % 详解: 赋值：计算表达式并保存到 z
z0 = 0;  % 详解: 赋值：计算表达式并保存到 z0
W = sqrt(N - length(S) - 3)*(z-z0);  % 详解: 赋值：将 sqrt(...) 的结果保存到 W
cutoff = norminv(1 - 0.5*alpha);  % 详解: 赋值：将 norminv(...) 的结果保存到 cutoff
if abs(W) < cutoff  % 详解: 条件判断：if (abs(W) < cutoff)
  CI = 1;  % 详解: 赋值：计算表达式并保存到 CI
else  % 详解: 条件判断：else 分支
  CI = 0;  % 详解: 赋值：计算表达式并保存到 CI
end  % 详解: 执行语句
p = normcdf(W);  % 详解: 赋值：将 normcdf(...) 的结果保存到 p


function p = normcdf(x,mu,sigma)  % 详解: 执行语句



if nargin < 3,  % 详解: 条件判断：if (nargin < 3,)
    sigma = 1;  % 详解: 赋值：计算表达式并保存到 sigma
end  % 详解: 执行语句

if nargin < 2;  % 详解: 条件判断：if (nargin < 2;)
    mu = 0;  % 详解: 赋值：计算表达式并保存到 mu
end  % 详解: 执行语句

[errorcode x mu sigma] = distchck(3,x,mu,sigma);  % 详解: 执行语句

if errorcode > 0  % 详解: 条件判断：if (errorcode > 0)
    error('Requires non-scalar arguments to match in size.');  % 详解: 调用函数：error('Requires non-scalar arguments to match in size.')
end  % 详解: 执行语句

p = zeros(size(x));  % 详解: 赋值：将 zeros(...) 的结果保存到 p

k1 = find(sigma <= 0);  % 详解: 赋值：将 find(...) 的结果保存到 k1
if any(k1)  % 详解: 条件判断：if (any(k1))
    tmp   = NaN;  % 详解: 赋值：计算表达式并保存到 tmp
    p(k1) = tmp(ones(size(k1)));  % 详解: 调用函数：p(k1) = tmp(ones(size(k1)))
end  % 详解: 执行语句

k = find(sigma > 0);  % 详解: 赋值：将 find(...) 的结果保存到 k
if any(k)  % 详解: 条件判断：if (any(k))
    p(k) = 0.5 * erfc( - (x(k) - mu(k)) ./ (sigma(k) * sqrt(2)));  % 详解: 调用函数：p(k) = 0.5 * erfc( - (x(k) - mu(k)) ./ (sigma(k) * sqrt(2)))
end  % 详解: 执行语句

k2 = find(p > 1);  % 详解: 赋值：将 find(...) 的结果保存到 k2
if any(k2)  % 详解: 条件判断：if (any(k2))
    p(k2) = ones(size(k2));  % 详解: 调用函数：p(k2) = ones(size(k2))
end  % 详解: 执行语句


function x = norminv(p,mu,sigma);  % 详解: 执行语句



if nargin < 3,  % 详解: 条件判断：if (nargin < 3,)
    sigma = 1;  % 详解: 赋值：计算表达式并保存到 sigma
end  % 详解: 执行语句

if nargin < 2;  % 详解: 条件判断：if (nargin < 2;)
    mu = 0;  % 详解: 赋值：计算表达式并保存到 mu
end  % 详解: 执行语句

[errorcode p mu sigma] = distchck(3,p,mu,sigma);  % 详解: 执行语句

if errorcode > 0  % 详解: 条件判断：if (errorcode > 0)
    error('Requires non-scalar arguments to match in size.');  % 详解: 调用函数：error('Requires non-scalar arguments to match in size.')
end  % 详解: 执行语句

x = zeros(size(p));  % 详解: 赋值：将 zeros(...) 的结果保存到 x

k = find(sigma <= 0 | p < 0 | p > 1);  % 详解: 赋值：将 find(...) 的结果保存到 k
if any(k)  % 详解: 条件判断：if (any(k))
    tmp  = NaN;  % 详解: 赋值：计算表达式并保存到 tmp
    x(k) = tmp(ones(size(k)));  % 详解: 调用函数：x(k) = tmp(ones(size(k)))
end  % 详解: 执行语句

k = find(p == 0);  % 详解: 赋值：将 find(...) 的结果保存到 k
if any(k)  % 详解: 条件判断：if (any(k))
    tmp  = Inf;  % 详解: 赋值：计算表达式并保存到 tmp
    x(k) = -tmp(ones(size(k)));  % 详解: 调用函数：x(k) = -tmp(ones(size(k)))
end  % 详解: 执行语句

k = find(p == 1);  % 详解: 赋值：将 find(...) 的结果保存到 k
if any(k)  % 详解: 条件判断：if (any(k))
    tmp  = Inf;  % 详解: 赋值：计算表达式并保存到 tmp
    x(k) = tmp(ones(size(k)));  % 详解: 调用函数：x(k) = tmp(ones(size(k)))
end  % 详解: 执行语句

k = find(p > 0  &  p < 1 & sigma > 0);  % 详解: 赋值：将 find(...) 的结果保存到 k
if any(k),  % 详解: 条件判断：if (any(k),)
    x(k) = sqrt(2) * sigma(k) .* erfinv(2 * p(k) - 1) + mu(k);  % 详解: 调用函数：x(k) = sqrt(2) * sigma(k) .* erfinv(2 * p(k) - 1) + mu(k)
end  % 详解: 执行语句




