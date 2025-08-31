% 文件: dhmm_em_online.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [transmat, obsmat, exp_num_trans, exp_num_emit, gamma, ll] = dhmm_em_online(...  % 详解: 执行语句
    prior, transmat, obsmat, exp_num_trans, exp_num_emit, decay, data, ...  % 详解: 执行语句
    act, adj_trans, adj_obs, dirichlet, filter_only)  % 详解: 执行语句

if ~exist('act'), act = []; end  % 详解: 条件判断：if (~exist('act'), act = []; end)
if ~exist('adj_trans'), adj_trans = 1; end  % 详解: 条件判断：if (~exist('adj_trans'), adj_trans = 1; end)
if ~exist('adj_obs'), adj_obs = 1; end  % 详解: 条件判断：if (~exist('adj_obs'), adj_obs = 1; end)
if ~exist('dirichlet'), dirichlet = 0; end  % 详解: 条件判断：if (~exist('dirichlet'), dirichlet = 0; end)
if ~exist('filter_only'), filter_only = 0; end  % 详解: 条件判断：if (~exist('filter_only'), filter_only = 0; end)

olikseq = multinomial_prob(data, obsmat);  % 详解: 赋值：将 multinomial_prob(...) 的结果保存到 olikseq
if isempty(act)  % 详解: 条件判断：if (isempty(act))
  [alpha, beta, gamma, ll, xi] = fwdback(prior, transmat, olikseq, 'fwd_only', filter_only);  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  [alpha, beta, gamma, ll, xi] = fwdback(prior, transmat, olikseq, 'fwd_only', filter_only, ...  % 详解: 执行语句
					 'act', act);  % 详解: 执行语句
end  % 详解: 执行语句

[S O] = size(obsmat);  % 详解: 获取向量/矩阵尺寸
if adj_obs  % 详解: 条件判断：if (adj_obs)
  exp_num_emit = decay*exp_num_emit + dirichlet*ones(S,O);  % 详解: 赋值：计算表达式并保存到 exp_num_emit
  T = length(data);  % 详解: 赋值：将 length(...) 的结果保存到 T
  if T < O  % 详解: 条件判断：if (T < O)
    for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
      o = data(t);  % 详解: 赋值：将 data(...) 的结果保存到 o
      exp_num_emit(:,o) = exp_num_emit(:,o) + gamma(:,t);  % 详解: 调用函数：exp_num_emit(:,o) = exp_num_emit(:,o) + gamma(:,t)
    end  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    for o=1:O  % 详解: for 循环：迭代变量 o 遍历 1:O
      ndx = find(data==o);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
      if ~isempty(ndx)  % 详解: 条件判断：if (~isempty(ndx))
	exp_num_emit(:,o) = exp_num_emit(:,o) + sum(gamma(:, ndx), 2);  % 详解: 调用函数：exp_num_emit(:,o) = exp_num_emit(:,o) + sum(gamma(:, ndx), 2)
      end  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

if adj_trans & (T > 1)  % 详解: 条件判断：if (adj_trans & (T > 1))
  if isempty(act)  % 详解: 条件判断：if (isempty(act))
    exp_num_trans = decay*exp_num_trans + sum(xi,3);  % 详解: 赋值：计算表达式并保存到 exp_num_trans
  else  % 详解: 条件判断：else 分支
    A = length(transmat);  % 详解: 赋值：将 length(...) 的结果保存到 A
    for a=1:A  % 详解: for 循环：迭代变量 a 遍历 1:A
      ndx = find(act(2:end)==a);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
      if ~isempty(ndx)  % 详解: 条件判断：if (~isempty(ndx))
	exp_num_trans{a} = decay*exp_num_trans{a} + sum(xi(:,:,ndx), 3);  % 详解: 统计：求和/均值/中位数
      end  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句



if adj_obs  % 详解: 条件判断：if (adj_obs)
  obsmat = mk_stochastic(exp_num_emit);  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 obsmat
end  % 详解: 执行语句
if adj_trans & (T>1)  % 详解: 条件判断：if (adj_trans & (T>1))
  if isempty(act)  % 详解: 条件判断：if (isempty(act))
    transmat = mk_stochastic(exp_num_trans);  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat
  else  % 详解: 条件判断：else 分支
    for a=1:A  % 详解: for 循环：迭代变量 a 遍历 1:A
      transmat{a} = mk_stochastic(exp_num_trans{a});  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句




