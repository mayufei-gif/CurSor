% 文件: dhmm_logprob_brute_force.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function logp = enumerate_HMM_loglik(prior, transmat, obsmat)  % 详解: 执行语句

Q = length(prior);  % 详解: 赋值：将 length(...) 的结果保存到 Q
T = size(obsmat, 2);  % 详解: 赋值：将 size(...) 的结果保存到 T
sizes = repmat(Q, 1, T);  % 详解: 赋值：将 repmat(...) 的结果保存到 sizes

psum = 0;  % 详解: 赋值：计算表达式并保存到 psum
for i=1:Q^T  % 详解: for 循环：迭代变量 i 遍历 1:Q^T
  qs = ind2subv(sizes, i);  % 详解: 赋值：将 ind2subv(...) 的结果保存到 qs
  psum = psum + prob_path(prior, transmat, obsmat, qs);  % 详解: 赋值：计算表达式并保存到 psum
end  % 详解: 执行语句
logp = log(psum)  % 详解: 赋值：将 log(...) 的结果保存到 logp

 




