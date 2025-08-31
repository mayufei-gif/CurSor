% 文件: dhmm_em_demo1.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

O = 3;  % 详解: 赋值：计算表达式并保存到 O
Q = 2;  % 详解: 赋值：计算表达式并保存到 Q

prior0 = normalise(rand(Q,1));  % 详解: 赋值：将 normalise(...) 的结果保存到 prior0
transmat0 = mk_stochastic(rand(Q,Q));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat0
obsmat0 = mk_stochastic(rand(Q,O));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 obsmat0

T = 5;  % 详解: 赋值：计算表达式并保存到 T
nex = 10;  % 详解: 赋值：计算表达式并保存到 nex
data = dhmm_sample(prior0, transmat0, obsmat0, T, nex);  % 详解: 赋值：将 dhmm_sample(...) 的结果保存到 data

prior1 = normalise(rand(Q,1));  % 详解: 赋值：将 normalise(...) 的结果保存到 prior1
transmat1 = mk_stochastic(rand(Q,Q));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat1
obsmat1 = mk_stochastic(rand(Q,O));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 obsmat1

[LL, prior2, transmat2, obsmat2] = dhmm_em(data(1,:), prior1, transmat1, obsmat1, 'max_iter', 5);  % 详解: 执行语句
LL  % 详解: 执行语句

loglik = dhmm_logprob(data(4,:), prior2, transmat2, obsmat2)  % 详解: 赋值：将 dhmm_logprob(...) 的结果保存到 loglik




