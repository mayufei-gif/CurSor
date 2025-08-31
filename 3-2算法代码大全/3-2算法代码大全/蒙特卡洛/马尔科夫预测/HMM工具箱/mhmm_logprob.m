% 文件: mhmm_logprob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [loglik, errors] = mhmm_logprob(data, prior, transmat, mu, Sigma, mixmat)  % 详解: 函数定义：mhmm_logprob(data, prior, transmat, mu, Sigma, mixmat), 返回：loglik, errors

Q = length(prior);  % 详解: 赋值：将 length(...) 的结果保存到 Q
if size(mixmat,1) ~= Q  % 详解: 条件判断：if (size(mixmat,1) ~= Q)
  error('mixmat should be QxM')  % 详解: 调用函数：error('mixmat should be QxM')
end  % 详解: 执行语句
if nargin < 6, mixmat = ones(Q,1); end  % 详解: 条件判断：if (nargin < 6, mixmat = ones(Q,1); end)

if ~iscell(data)  % 详解: 条件判断：if (~iscell(data))
  data = num2cell(data, [1 2]);  % 详解: 赋值：将 num2cell(...) 的结果保存到 data
end  % 详解: 执行语句
ncases = length(data);  % 详解: 赋值：将 length(...) 的结果保存到 ncases

loglik = 0;  % 详解: 赋值：计算表达式并保存到 loglik
errors = [];  % 详解: 赋值：计算表达式并保存到 errors
for m=1:ncases  % 详解: for 循环：迭代变量 m 遍历 1:ncases
  obslik = mixgauss_prob(data{m}, mu, Sigma, mixmat);  % 详解: 赋值：将 mixgauss_prob(...) 的结果保存到 obslik
  [alpha, beta, gamma, ll] = fwdback(prior, transmat, obslik, 'fwd_only', 1);  % 详解: 执行语句
  if ll==-inf  % 详解: 条件判断：if (ll==-inf)
    errors = [errors m];  % 详解: 赋值：计算表达式并保存到 errors
  end  % 详解: 执行语句
  loglik = loglik + ll;  % 详解: 赋值：计算表达式并保存到 loglik
end  % 详解: 执行语句




