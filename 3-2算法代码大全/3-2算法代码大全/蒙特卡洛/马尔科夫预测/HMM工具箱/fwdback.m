% 文件: fwdback.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [alpha, beta, gamma, loglik, xi_summed, gamma2] = fwdback(init_state_distrib, ...  % 详解: 执行语句
   transmat, obslik, varargin)  % 详解: 执行语句

if 0  % 详解: 条件判断：if (0)
  warning('this now returns sum_t xi(i,j,t) not xi(i,j,t)')  % 详解: 调用函数：warning('this now returns sum_t xi(i,j,t) not xi(i,j,t)')
end  % 详解: 执行语句

if nargout >= 5, compute_xi = 1; else compute_xi = 0; end  % 详解: 条件判断：if (nargout >= 5, compute_xi = 1; else compute_xi = 0; end)
if nargout >= 6, compute_gamma2 = 1; else compute_gamma2 = 0; end  % 详解: 条件判断：if (nargout >= 6, compute_gamma2 = 1; else compute_gamma2 = 0; end)

[obslik2, mixmat, fwd_only, scaled, act, maximize, compute_xi, compute_gamma2] = ...  % 详解: 执行语句
   process_options(varargin, ...  % 详解: 执行语句
       'obslik2', [], 'mixmat', [], ...  % 详解: 执行语句
       'fwd_only', 0, 'scaled', 1, 'act', [], 'maximize', 0, ...  % 详解: 执行语句
                   'compute_xi', compute_xi, 'compute_gamma2', compute_gamma2);  % 详解: 执行语句

[Q T] = size(obslik);  % 详解: 获取向量/矩阵尺寸

if isempty(obslik2)  % 详解: 条件判断：if (isempty(obslik2))
 compute_gamma2 = 0;  % 详解: 赋值：计算表达式并保存到 compute_gamma2
end  % 详解: 执行语句

if isempty(act)  % 详解: 条件判断：if (isempty(act))
 act = ones(1,T);  % 详解: 赋值：将 ones(...) 的结果保存到 act
 transmat = { transmat } ;  % 详解: 赋值：计算表达式并保存到 transmat
end  % 详解: 执行语句

scale = ones(1,T);  % 详解: 赋值：将 ones(...) 的结果保存到 scale


loglik = 0;  % 详解: 赋值：计算表达式并保存到 loglik

alpha = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 alpha
gamma = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 gamma
if compute_xi  % 详解: 条件判断：if (compute_xi)
 xi_summed = zeros(Q,Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 xi_summed
else  % 详解: 条件判断：else 分支
 xi_summed = [];  % 详解: 赋值：计算表达式并保存到 xi_summed
end  % 详解: 执行语句


t = 1;  % 详解: 赋值：计算表达式并保存到 t
alpha(:,1) = init_state_distrib(:) .* obslik(:,t);  % 详解: 调用函数：alpha(:,1) = init_state_distrib(:) .* obslik(:,t)
if scaled  % 详解: 条件判断：if (scaled)
 [alpha(:,t), scale(t)] = normalise(alpha(:,t));  % 详解: 执行语句
end  % 详解: 执行语句
for t=2:T  % 详解: for 循环：迭代变量 t 遍历 2:T
 trans = transmat{act(t-1)};  % 详解: 赋值：计算表达式并保存到 trans
 if maximize  % 详解: 条件判断：if (maximize)
   m = max_mult(trans', alpha(:,t-1));  % 赋值：设置变量 m  % 详解: 赋值：将 max_mult(...) 的结果保存到 m  % 详解: 赋值：将 max_mult(...) 的结果保存到 m
 else  % 详解: 条件判断：else 分支
   m = trans' * alpha(:,t-1);  % 赋值：设置变量 m  % 详解: 赋值：计算表达式并保存到 m  % 详解: 赋值：计算表达式并保存到 m
 end  % 详解: 执行语句
 alpha(:,t) = m(:) .* obslik(:,t);  % 详解: 调用函数：alpha(:,t) = m(:) .* obslik(:,t)
 if scaled  % 详解: 条件判断：if (scaled)
   [alpha(:,t), scale(t)] = normalise(alpha(:,t));  % 详解: 执行语句
 end  % 详解: 执行语句
 if compute_xi & fwd_only  % 详解: 条件判断：if (compute_xi & fwd_only)
   xi_summed = xi_summed + normalise((alpha(:,t-1) * obslik(:,t)') .* trans);  % 赋值：设置变量 xi_summed  % 详解: 赋值：计算表达式并保存到 xi_summed  % 详解: 赋值：计算表达式并保存到 xi_summed
 end  % 详解: 执行语句
end  % 详解: 执行语句
if scaled  % 详解: 条件判断：if (scaled)
 if any(scale==0)  % 详解: 条件判断：if (any(scale==0))
   loglik = -inf;  % 详解: 赋值：计算表达式并保存到 loglik
 else  % 详解: 条件判断：else 分支
   loglik = sum(log(scale));  % 详解: 赋值：将 sum(...) 的结果保存到 loglik
 end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
 loglik = log(sum(alpha(:,T)));  % 详解: 赋值：将 log(...) 的结果保存到 loglik
end  % 详解: 执行语句

if fwd_only  % 详解: 条件判断：if (fwd_only)
 gamma = alpha;  % 详解: 赋值：计算表达式并保存到 gamma
 beta = [];  % 详解: 赋值：计算表达式并保存到 beta
 gamma2 = [];  % 详解: 赋值：计算表达式并保存到 gamma2
 return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句


beta = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 beta
if compute_gamma2  % 详解: 条件判断：if (compute_gamma2)
  if iscell(mixmat)  % 详解: 条件判断：if (iscell(mixmat))
    M = size(mixmat{1},2);  % 详解: 赋值：将 size(...) 的结果保存到 M
  else  % 详解: 条件判断：else 分支
    M = size(mixmat, 2);  % 详解: 赋值：将 size(...) 的结果保存到 M
  end  % 详解: 执行语句
 gamma2 = zeros(Q,M,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 gamma2
else  % 详解: 条件判断：else 分支
 gamma2 = [];  % 详解: 赋值：计算表达式并保存到 gamma2
end  % 详解: 执行语句

beta(:,T) = ones(Q,1);  % 详解: 调用函数：beta(:,T) = ones(Q,1)
gamma(:,T) = normalise(alpha(:,T) .* beta(:,T));  % 详解: 调用函数：gamma(:,T) = normalise(alpha(:,T) .* beta(:,T))
t=T;  % 详解: 赋值：计算表达式并保存到 t
if compute_gamma2  % 详解: 条件判断：if (compute_gamma2)
 denom = obslik(:,t) + (obslik(:,t)==0);  % 详解: 赋值：将 obslik(...) 的结果保存到 denom
 if iscell(mixmat)  % 详解: 条件判断：if (iscell(mixmat))
   gamma2(:,:,t) = obslik2(:,:,t) .* mixmat{t} .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M]);  % 详解: 调用函数：gamma2(:,:,t) = obslik2(:,:,t) .* mixmat{t} .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M])
 else  % 详解: 条件判断：else 分支
   gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M]);  % 详解: 调用函数：gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M])
 end  % 详解: 执行语句
end  % 详解: 执行语句
for t=T-1:-1:1  % 详解: for 循环：迭代变量 t 遍历 T-1:-1:1
 b = beta(:,t+1) .* obslik(:,t+1);  % 详解: 赋值：将 beta(...) 的结果保存到 b
 trans = transmat{act(t)};  % 详解: 赋值：计算表达式并保存到 trans
 if maximize  % 详解: 条件判断：if (maximize)
   B = repmat(b(:)', Q, 1);  % 赋值：设置变量 B  % 详解: 赋值：将 repmat(...) 的结果保存到 B  % 详解: 赋值：将 repmat(...) 的结果保存到 B
   beta(:,t) = max(trans .* B, [], 2);  % 详解: 调用函数：beta(:,t) = max(trans .* B, [], 2)
 else  % 详解: 条件判断：else 分支
   beta(:,t) = trans * b;  % 详解: 执行语句
 end  % 详解: 执行语句
 if scaled  % 详解: 条件判断：if (scaled)
   beta(:,t) = normalise(beta(:,t));  % 详解: 调用函数：beta(:,t) = normalise(beta(:,t))
 end  % 详解: 执行语句
 gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));  % 详解: 调用函数：gamma(:,t) = normalise(alpha(:,t) .* beta(:,t))
 if compute_xi  % 详解: 条件判断：if (compute_xi)
   xi_summed = xi_summed + normalise((trans .* (alpha(:,t) * b')));  % 赋值：设置变量 xi_summed  % 详解: 赋值：计算表达式并保存到 xi_summed  % 详解: 赋值：计算表达式并保存到 xi_summed
 end  % 详解: 执行语句
 if compute_gamma2  % 详解: 条件判断：if (compute_gamma2)
   denom = obslik(:,t) + (obslik(:,t)==0);  % 详解: 赋值：将 obslik(...) 的结果保存到 denom
   if iscell(mixmat)  % 详解: 条件判断：if (iscell(mixmat))
     gamma2(:,:,t) = obslik2(:,:,t) .* mixmat{t} .* repmat(gamma(:,t), [1 M]) ./ repmat(denom,  [1 M]);  % 详解: 调用函数：gamma2(:,:,t) = obslik2(:,:,t) .* mixmat{t} .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M])
   else  % 详解: 条件判断：else 分支
     gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom,  [1 M]);  % 详解: 调用函数：gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom, [1 M])
   end  % 详解: 执行语句
 end  % 详解: 执行语句
end  % 详解: 执行语句





