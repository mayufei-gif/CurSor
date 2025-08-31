% 文件: parzenC_test.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

d = 2; M = 3; Q = 4; T = 5; Sigma = 10;  % 详解: 赋值：计算表达式并保存到 d
N = sample_discrete(normalize(ones(1,M)), 1, Q);  % 详解: 赋值：将 sample_discrete(...) 的结果保存到 N
data = randn(d,T);  % 详解: 赋值：将 randn(...) 的结果保存到 data
mu = randn(d,M,Q);  % 详解: 赋值：将 randn(...) 的结果保存到 mu

[BM, B2M] = parzen(data, mu, Sigma, N);  % 详解: 执行语句
[B, B2] = parzenC(data, mu, Sigma, N);  % 详解: 执行语句

approxeq(B,BM)  % 详解: 调用函数：approxeq(B,BM)
approxeq(B2,B2M)  % 详解: 调用函数：approxeq(B2,B2M)




