% 文件: dhmm_em.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [LL, prior, transmat, obsmat, nrIterations] = ...  % 详解: 执行语句
   dhmm_em(data, prior, transmat, obsmat, varargin)  % 详解: 调用函数：dhmm_em(data, prior, transmat, obsmat, varargin)

[max_iter, thresh, verbose, obs_prior_weight, adj_prior, adj_trans, adj_obs] = ...  % 详解: 执行语句
   process_options(varargin, 'max_iter', 10, 'thresh', 1e-4, 'verbose', 1, ...  % 详解: 执行语句
                   'obs_prior_weight', 0, 'adj_prior', 1, 'adj_trans', 1, 'adj_obs', 1);  % 详解: 执行语句

previous_loglik = -inf;  % 详解: 赋值：计算表达式并保存到 previous_loglik
loglik = 0;  % 详解: 赋值：计算表达式并保存到 loglik
converged = 0;  % 详解: 赋值：计算表达式并保存到 converged
num_iter = 1;  % 详解: 赋值：计算表达式并保存到 num_iter
LL = [];  % 详解: 赋值：计算表达式并保存到 LL

if ~iscell(data)  % 详解: 条件判断：if (~iscell(data))
 data = num2cell(data, 2);  % 详解: 赋值：将 num2cell(...) 的结果保存到 data
end  % 详解: 执行语句

while (num_iter <= max_iter) & ~converged  % 详解: while 循环：当 ((num_iter <= max_iter) & ~converged) 为真时迭代
 [loglik, exp_num_trans, exp_num_visits1, exp_num_emit] = ...  % 详解: 执行语句
     compute_ess_dhmm(prior, transmat, obsmat, data, obs_prior_weight);  % 详解: 调用函数：compute_ess_dhmm(prior, transmat, obsmat, data, obs_prior_weight)

 if adj_prior  % 详解: 条件判断：if (adj_prior)
   prior = normalise(exp_num_visits1);  % 详解: 赋值：将 normalise(...) 的结果保存到 prior
 end  % 详解: 执行语句
 if adj_trans & ~isempty(exp_num_trans)  % 详解: 条件判断：if (adj_trans & ~isempty(exp_num_trans))
   transmat = mk_stochastic(exp_num_trans);  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat
 end  % 详解: 执行语句
 if adj_obs  % 详解: 条件判断：if (adj_obs)
   obsmat = mk_stochastic(exp_num_emit);  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 obsmat
 end  % 详解: 执行语句

 if verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end  % 详解: 条件判断：if (verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end)
 num_iter =  num_iter + 1;  % 详解: 赋值：计算表达式并保存到 num_iter
 converged = em_converged(loglik, previous_loglik, thresh);  % 详解: 赋值：将 em_converged(...) 的结果保存到 converged
 previous_loglik = loglik;  % 详解: 赋值：计算表达式并保存到 previous_loglik
 LL = [LL loglik];  % 详解: 赋值：计算表达式并保存到 LL
end  % 详解: 执行语句
nrIterations = num_iter - 1;  % 详解: 赋值：计算表达式并保存到 nrIterations


function [loglik, exp_num_trans, exp_num_visits1, exp_num_emit, exp_num_visitsT] = ...  % 详解: 执行语句
   compute_ess_dhmm(startprob, transmat, obsmat, data, dirichlet)  % 详解: 调用函数：compute_ess_dhmm(startprob, transmat, obsmat, data, dirichlet)

numex = length(data);  % 详解: 赋值：将 length(...) 的结果保存到 numex
[S O] = size(obsmat);  % 详解: 获取向量/矩阵尺寸
exp_num_trans = zeros(S,S);  % 详解: 赋值：将 zeros(...) 的结果保存到 exp_num_trans
exp_num_visits1 = zeros(S,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 exp_num_visits1
exp_num_visitsT = zeros(S,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 exp_num_visitsT
exp_num_emit = dirichlet*ones(S,O);  % 详解: 赋值：计算表达式并保存到 exp_num_emit
loglik = 0;  % 详解: 赋值：计算表达式并保存到 loglik

for ex=1:numex  % 详解: for 循环：迭代变量 ex 遍历 1:numex
 obs = data{ex};  % 详解: 赋值：计算表达式并保存到 obs
 T = length(obs);  % 详解: 赋值：将 length(...) 的结果保存到 T
 obslik = multinomial_prob(obs, obsmat);  % 详解: 赋值：将 multinomial_prob(...) 的结果保存到 obslik
 [alpha, beta, gamma, current_ll, xi_summed] = fwdback(startprob, transmat, obslik);  % 详解: 执行语句

 loglik = loglik +  current_ll;  % 详解: 赋值：计算表达式并保存到 loglik
 exp_num_trans = exp_num_trans + xi_summed;  % 详解: 赋值：计算表达式并保存到 exp_num_trans
 exp_num_visits1 = exp_num_visits1 + gamma(:,1);  % 详解: 赋值：计算表达式并保存到 exp_num_visits1
 exp_num_visitsT = exp_num_visitsT + gamma(:,T);  % 详解: 赋值：计算表达式并保存到 exp_num_visitsT
 if T < O  % 详解: 条件判断：if (T < O)
   for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
     o = obs(t);  % 详解: 赋值：将 obs(...) 的结果保存到 o
     exp_num_emit(:,o) = exp_num_emit(:,o) + gamma(:,t);  % 详解: 调用函数：exp_num_emit(:,o) = exp_num_emit(:,o) + gamma(:,t)
   end  % 详解: 执行语句
 else  % 详解: 条件判断：else 分支
   for o=1:O  % 详解: for 循环：迭代变量 o 遍历 1:O
     ndx = find(obs==o);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
     if ~isempty(ndx)  % 详解: 条件判断：if (~isempty(ndx))
       exp_num_emit(:,o) = exp_num_emit(:,o) + sum(gamma(:, ndx), 2);  % 详解: 调用函数：exp_num_emit(:,o) = exp_num_emit(:,o) + sum(gamma(:, ndx), 2)
     end  % 详解: 执行语句
   end  % 详解: 执行语句
 end  % 详解: 执行语句
end  % 详解: 执行语句




