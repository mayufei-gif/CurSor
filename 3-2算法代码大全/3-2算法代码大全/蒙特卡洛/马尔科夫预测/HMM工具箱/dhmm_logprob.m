% 文件: dhmm_logprob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [loglik, errors] = dhmm_logprob(data, prior, transmat, obsmat)  % 详解: 函数定义：dhmm_logprob(data, prior, transmat, obsmat), 返回：loglik, errors

if ~iscell(data)  % 详解: 条件判断：if (~iscell(data))
  data = num2cell(data, 2);  % 详解: 赋值：将 num2cell(...) 的结果保存到 data
end  % 详解: 执行语句
ncases = length(data);  % 详解: 赋值：将 length(...) 的结果保存到 ncases

loglik = 0;  % 详解: 赋值：计算表达式并保存到 loglik
errors = [];  % 详解: 赋值：计算表达式并保存到 errors
for m=1:ncases  % 详解: for 循环：迭代变量 m 遍历 1:ncases
  obslik = multinomial_prob(data{m}, obsmat);  % 详解: 赋值：将 multinomial_prob(...) 的结果保存到 obslik
  [alpha, beta, gamma, ll] = fwdback(prior, transmat, obslik, 'fwd_only', 1);  % 详解: 执行语句
  if ll==-inf  % 详解: 条件判断：if (ll==-inf)
    errors = [errors m];  % 详解: 赋值：计算表达式并保存到 errors
  end  % 详解: 执行语句
  loglik = loglik + ll;  % 详解: 赋值：计算表达式并保存到 loglik
end  % 详解: 执行语句




