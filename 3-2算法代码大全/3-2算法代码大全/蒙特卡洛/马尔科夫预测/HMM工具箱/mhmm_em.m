% 文件: mhmm_em.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [LL, prior, transmat, mu, Sigma, mixmat] = ...  % 详解: 执行语句
     mhmm_em(data, prior, transmat, mu, Sigma, mixmat, varargin);  % 详解: 调用函数：mhmm_em(data, prior, transmat, mu, Sigma, mixmat, varargin)

if ~isempty(varargin) & ~isstr(varargin{1})  % 详解: 条件判断：if (~isempty(varargin) & ~isstr(varargin{1}))
  error('optional arguments should be passed as string/value pairs')  % 详解: 调用函数：error('optional arguments should be passed as string/value pairs')
end  % 详解: 执行语句

[max_iter, thresh, verbose, cov_type,  adj_prior, adj_trans, adj_mix, adj_mu, adj_Sigma] = ...  % 详解: 执行语句
    process_options(varargin, 'max_iter', 10, 'thresh', 1e-4, 'verbose', 1, ...  % 详解: 执行语句
		    'cov_type', 'full', 'adj_prior', 1, 'adj_trans', 1, 'adj_mix', 1, ...  % 详解: 执行语句
		    'adj_mu', 1, 'adj_Sigma', 1);  % 详解: 执行语句
  
previous_loglik = -inf;  % 详解: 赋值：计算表达式并保存到 previous_loglik
loglik = 0;  % 详解: 赋值：计算表达式并保存到 loglik
converged = 0;  % 详解: 赋值：计算表达式并保存到 converged
num_iter = 1;  % 详解: 赋值：计算表达式并保存到 num_iter
LL = [];  % 详解: 赋值：计算表达式并保存到 LL

if ~iscell(data)  % 详解: 条件判断：if (~iscell(data))
  data = num2cell(data, [1 2]);  % 详解: 赋值：将 num2cell(...) 的结果保存到 data
end  % 详解: 执行语句
numex = length(data);  % 详解: 赋值：将 length(...) 的结果保存到 numex


O = size(data{1},1);  % 详解: 赋值：将 size(...) 的结果保存到 O
Q = length(prior);  % 详解: 赋值：将 length(...) 的结果保存到 Q
if isempty(mixmat)  % 详解: 条件判断：if (isempty(mixmat))
  mixmat = ones(Q,1);  % 详解: 赋值：将 ones(...) 的结果保存到 mixmat
end  % 详解: 执行语句
M = size(mixmat,2);  % 详解: 赋值：将 size(...) 的结果保存到 M
if M == 1  % 详解: 条件判断：if (M == 1)
  adj_mix = 0;  % 详解: 赋值：计算表达式并保存到 adj_mix
end  % 详解: 执行语句

while (num_iter <= max_iter) & ~converged  % 详解: while 循环：当 ((num_iter <= max_iter) & ~converged) 为真时迭代
  [loglik, exp_num_trans, exp_num_visits1, postmix, m, ip, op] = ...  % 详解: 执行语句
      ess_mhmm(prior, transmat, mixmat, mu, Sigma, data);  % 详解: 调用函数：ess_mhmm(prior, transmat, mixmat, mu, Sigma, data)
  
  
  if adj_prior  % 详解: 条件判断：if (adj_prior)
    prior = normalise(exp_num_visits1);  % 详解: 赋值：将 normalise(...) 的结果保存到 prior
  end  % 详解: 执行语句
  if adj_trans  % 详解: 条件判断：if (adj_trans)
    transmat = mk_stochastic(exp_num_trans);  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat
  end  % 详解: 执行语句
  if adj_mix  % 详解: 条件判断：if (adj_mix)
    mixmat = mk_stochastic(postmix);  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 mixmat
  end  % 详解: 执行语句
  if adj_mu | adj_Sigma  % 详解: 条件判断：if (adj_mu | adj_Sigma)
    [mu2, Sigma2] = mixgauss_Mstep(postmix, m, op, ip, 'cov_type', cov_type);  % 详解: 执行语句
    if adj_mu  % 详解: 条件判断：if (adj_mu)
      mu = reshape(mu2, [O Q M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu
    end  % 详解: 执行语句
    if adj_Sigma  % 详解: 条件判断：if (adj_Sigma)
      Sigma = reshape(Sigma2, [O O Q M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 Sigma
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  
  if verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end  % 详解: 条件判断：if (verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end)
  num_iter =  num_iter + 1;  % 详解: 赋值：计算表达式并保存到 num_iter
  converged = em_converged(loglik, previous_loglik, thresh);  % 详解: 赋值：将 em_converged(...) 的结果保存到 converged
  previous_loglik = loglik;  % 详解: 赋值：计算表达式并保存到 previous_loglik
  LL = [LL loglik];  % 详解: 赋值：计算表达式并保存到 LL
end  % 详解: 执行语句



function [loglik, exp_num_trans, exp_num_visits1, postmix, m, ip, op] = ...  % 详解: 执行语句
    ess_mhmm(prior, transmat, mixmat, mu, Sigma, data)  % 详解: 调用函数：ess_mhmm(prior, transmat, mixmat, mu, Sigma, data)


verbose = 0;  % 详解: 赋值：计算表达式并保存到 verbose

numex = length(data);  % 详解: 赋值：将 length(...) 的结果保存到 numex
O = size(data{1},1);  % 详解: 赋值：将 size(...) 的结果保存到 O
Q = length(prior);  % 详解: 赋值：将 length(...) 的结果保存到 Q
M = size(mixmat,2);  % 详解: 赋值：将 size(...) 的结果保存到 M
exp_num_trans = zeros(Q,Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 exp_num_trans
exp_num_visits1 = zeros(Q,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 exp_num_visits1
postmix = zeros(Q,M);  % 详解: 赋值：将 zeros(...) 的结果保存到 postmix
m = zeros(O,Q,M);  % 详解: 赋值：将 zeros(...) 的结果保存到 m
op = zeros(O,O,Q,M);  % 详解: 赋值：将 zeros(...) 的结果保存到 op
ip = zeros(Q,M);  % 详解: 赋值：将 zeros(...) 的结果保存到 ip

mix = (M>1);  % 详解: 赋值：计算表达式并保存到 mix

loglik = 0;  % 详解: 赋值：计算表达式并保存到 loglik
if verbose, fprintf(1, 'forwards-backwards example # '); end  % 详解: 条件判断：if (verbose, fprintf(1, 'forwards-backwards example # '); end)
for ex=1:numex  % 详解: for 循环：迭代变量 ex 遍历 1:numex
  if verbose, fprintf(1, '%d ', ex); end  % 详解: 条件判断：if (verbose, fprintf(1, '%d ', ex); end)
  obs = data{ex};  % 详解: 赋值：计算表达式并保存到 obs
  T = size(obs,2);  % 详解: 赋值：将 size(...) 的结果保存到 T
  if mix  % 详解: 条件判断：if (mix)
    [B, B2] = mixgauss_prob(obs, mu, Sigma, mixmat);  % 详解: 执行语句
    [alpha, beta, gamma,  current_loglik, xi_summed, gamma2] = ...  % 详解: 执行语句
	fwdback(prior, transmat, B, 'obslik2', B2, 'mixmat', mixmat);  % 详解: 调用函数：fwdback(prior, transmat, B, 'obslik2', B2, 'mixmat', mixmat)
  else  % 详解: 条件判断：else 分支
    B = mixgauss_prob(obs, mu, Sigma);  % 详解: 赋值：将 mixgauss_prob(...) 的结果保存到 B
    [alpha, beta, gamma,  current_loglik, xi_summed] = fwdback(prior, transmat, B);  % 详解: 执行语句
  end  % 详解: 执行语句
  loglik = loglik +  current_loglik;  % 详解: 赋值：计算表达式并保存到 loglik
  if verbose, fprintf(1, 'll at ex %d = %f\n', ex, loglik); end  % 详解: 条件判断：if (verbose, fprintf(1, 'll at ex %d = %f\n', ex, loglik); end)

  exp_num_trans = exp_num_trans + xi_summed;  % 详解: 赋值：计算表达式并保存到 exp_num_trans
  exp_num_visits1 = exp_num_visits1 + gamma(:,1);  % 详解: 赋值：计算表达式并保存到 exp_num_visits1
  
  if mix  % 详解: 条件判断：if (mix)
    postmix = postmix + sum(gamma2,3);  % 详解: 赋值：计算表达式并保存到 postmix
  else  % 详解: 条件判断：else 分支
    postmix = postmix + sum(gamma,2);  % 详解: 赋值：计算表达式并保存到 postmix
    gamma2 = reshape(gamma, [Q 1 T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 gamma2
  end  % 详解: 执行语句
  for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
    for k=1:M  % 详解: for 循环：迭代变量 k 遍历 1:M
      w = reshape(gamma2(i,k,:), [1 T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 w
      wobs = obs .* repmat(w, [O 1]);  % 详解: 赋值：计算表达式并保存到 wobs
      m(:,i,k) = m(:,i,k) + sum(wobs, 2);  % 详解: 调用函数：m(:,i,k) = m(:,i,k) + sum(wobs, 2)
      op(:,:,i,k) = op(:,:,i,k) + wobs * obs'; % op(:,:) = sum_t w(t) * obs(:,t) * obs(:,t)'  % 详解: 执行语句
      ip(i,k) = ip(i,k) + sum(sum(wobs .* obs, 2));  % 详解: 调用函数：ip(i,k) = ip(i,k) + sum(sum(wobs .* obs, 2))
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句
if verbose, fprintf(1, '\n'); end  % 详解: 条件判断：if (verbose, fprintf(1, '\n'); end)




