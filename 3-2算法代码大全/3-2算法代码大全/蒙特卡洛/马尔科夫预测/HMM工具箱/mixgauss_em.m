% 文件: mixgauss_em.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [mu, Sigma, prior] = mixgauss_em(Y, nc, varargin)  % 详解: 函数定义：mixgauss_em(Y, nc, varargin), 返回：mu, Sigma, prior


[max_iter, thresh, cov_type, mu, Sigma, method, ...  % 详解: 执行语句
 cov_prior, verbose, prune_thresh] = process_options(...  % 详解: 执行语句
    varargin, 'max_iter', 10, 'thresh', 1e-2, 'cov_type', 'full', ...  % 详解: 执行语句
    'mu', [], 'Sigma', [],  'method', 'kmeans', ...  % 详解: 执行语句
    'cov_prior', [], 'verbose', 0, 'prune_thresh', 0);  % 详解: 执行语句

[ny T] = size(Y);  % 详解: 获取向量/矩阵尺寸

if nc==1  % 详解: 条件判断：if (nc==1)
  mu = mean(Y')';  % 详解: 赋值：将 mean(...) 的结果保存到 mu
  Sigma = cov(Y');  % 赋值：设置变量 Sigma  % 详解: 赋值：将 cov(...) 的结果保存到 Sigma  % 详解: 赋值：将 cov(...) 的结果保存到 Sigma
  if strcmp(cov_type, 'diag')  % 详解: 条件判断：if (strcmp(cov_type, 'diag'))
    Sigma = diag(diag(Sigma));  % 详解: 赋值：将 diag(...) 的结果保存到 Sigma
  end  % 详解: 执行语句
  prior = 1;  % 详解: 赋值：计算表达式并保存到 prior
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

if isempty(mu)  % 详解: 条件判断：if (isempty(mu))
  [mu, Sigma, prior] = mixgauss_init(nc, Y, cov_type, method);  % 详解: 执行语句
end  % 详解: 执行语句

previous_loglik = -inf;  % 详解: 赋值：计算表达式并保存到 previous_loglik
num_iter = 1;  % 详解: 赋值：计算表达式并保存到 num_iter
converged = 0;  % 详解: 赋值：计算表达式并保存到 converged


while (num_iter <= max_iter) & ~converged  % 详解: while 循环：当 ((num_iter <= max_iter) & ~converged) 为真时迭代
  probY = mixgauss_prob(Y, mu, Sigma, prior);  % 详解: 赋值：将 mixgauss_prob(...) 的结果保存到 probY
  [post, lik] = normalize(probY .* repmat(prior, 1, T), 1);  % 详解: 执行语句
  loglik = log(sum(lik));  % 详解: 赋值：将 log(...) 的结果保存到 loglik
 
  w = sum(post,2);  % 详解: 赋值：将 sum(...) 的结果保存到 w
  WYY = zeros(ny, ny, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 WYY
  WY = zeros(ny, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 WY
  WYTY = zeros(nc,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 WYTY
  for c=1:nc  % 详解: for 循环：迭代变量 c 遍历 1:nc
    weights = repmat(post(c,:), ny, 1);  % 详解: 赋值：将 repmat(...) 的结果保存到 weights
    WYbig = Y .* weights;  % 详解: 赋值：计算表达式并保存到 WYbig
    WYY(:,:,c) = WYbig * Y';  % 调用函数：WYY  % 详解: 执行语句  % 详解: 执行语句
    WY(:,c) = sum(WYbig, 2);  % 详解: 调用函数：WY(:,c) = sum(WYbig, 2)
    WYTY(c) = sum(diag(WYbig' * Y));   % 统计：求和/均值/中位数  % 详解: 调用函数：WYTY(c) = sum(diag(WYbig' * Y))  % 详解: 调用函数：WYTY(c) = sum(diag(WYbig' * Y)); % 统计：求和/均值/中位数 % 详解: 调用函数：WYTY(c) = sum(diag(WYbig' * Y))
  end  % 详解: 执行语句
  
  prior = normalize(w);  % 详解: 赋值：将 normalize(...) 的结果保存到 prior
  [mu, Sigma] = mixgauss_Mstep(w, WY, WYY, WYTY, 'cov_type', cov_type, 'cov_prior', cov_prior);  % 详解: 执行语句
  
  if verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end  % 详解: 条件判断：if (verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end)
  num_iter =  num_iter + 1;  % 详解: 赋值：计算表达式并保存到 num_iter
  converged = em_converged(loglik, previous_loglik, thresh);  % 详解: 赋值：将 em_converged(...) 的结果保存到 converged
  previous_loglik = loglik;  % 详解: 赋值：计算表达式并保存到 previous_loglik
  
end  % 详解: 执行语句

if prune_thresh > 0  % 详解: 条件判断：if (prune_thresh > 0)
  ndx = find(prior < prune_thresh);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
  mu(:,ndx) = [];  % 详解: 执行语句
  Sigma(:,:,ndx) = [];  % 详解: 执行语句
  prior(ndx) = [];  % 详解: 执行语句
end  % 详解: 执行语句




