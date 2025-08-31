% 文件: mhmm_em_demo.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

if 1  % 详解: 条件判断：if (1)
  O = 4;  % 详解: 赋值：计算表达式并保存到 O
  T = 10;  % 详解: 赋值：计算表达式并保存到 T
  nex = 50;  % 详解: 赋值：计算表达式并保存到 nex
  M = 2;  % 详解: 赋值：计算表达式并保存到 M
  Q = 3;  % 详解: 赋值：计算表达式并保存到 Q
else  % 详解: 条件判断：else 分支
  O = 8;  % 详解: 赋值：计算表达式并保存到 O
  T = 420;  % 详解: 赋值：计算表达式并保存到 T
  nex = 1;  % 详解: 赋值：计算表达式并保存到 nex
  M = 1;  % 详解: 赋值：计算表达式并保存到 M
  Q = 6;  % 详解: 赋值：计算表达式并保存到 Q
end  % 详解: 执行语句
cov_type = 'full';  % 详解: 赋值：计算表达式并保存到 cov_type

data = randn(O,T,nex);  % 详解: 赋值：将 randn(...) 的结果保存到 data

prior0 = normalise(rand(Q,1));  % 详解: 赋值：将 normalise(...) 的结果保存到 prior0
transmat0 = mk_stochastic(rand(Q,Q));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat0

if 0  % 详解: 条件判断：if (0)
  Sigma0 = repmat(eye(O), [1 1 Q M]);  % 详解: 赋值：将 repmat(...) 的结果保存到 Sigma0
  indices = randperm(T*nex);  % 详解: 赋值：将 randperm(...) 的结果保存到 indices
  mu0 = reshape(data(:,indices(1:(Q*M))), [O Q M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu0
  mixmat0 = mk_stochastic(rand(Q,M));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 mixmat0
else  % 详解: 条件判断：else 分支
  [mu0, Sigma0] = mixgauss_init(Q*M, data, cov_type);  % 详解: 执行语句
  mu0 = reshape(mu0, [O Q M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu0
  Sigma0 = reshape(Sigma0, [O O Q M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 Sigma0
  mixmat0 = mk_stochastic(rand(Q,M));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 mixmat0
end  % 详解: 执行语句

[LL, prior1, transmat1, mu1, Sigma1, mixmat1] = ...  % 详解: 执行语句
    mhmm_em(data, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 5);  % 详解: 调用函数：mhmm_em(data, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 5)


loglik = mhmm_logprob(data, prior1, transmat1, mu1, Sigma1, mixmat1);  % 详解: 赋值：将 mhmm_logprob(...) 的结果保存到 loglik





