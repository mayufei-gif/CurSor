% 文件: mhmm_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [obs, hidden] = mhmm_sample(T, numex, initial_prob, transmat, mu, Sigma, mixmat)  % 详解: 函数定义：mhmm_sample(T, numex, initial_prob, transmat, mu, Sigma, mixmat), 返回：obs, hidden

Q = length(initial_prob);  % 详解: 赋值：将 length(...) 的结果保存到 Q
if nargin < 7, mixmat = ones(Q,1); end  % 详解: 条件判断：if (nargin < 7, mixmat = ones(Q,1); end)
O = size(mu,1);  % 详解: 赋值：将 size(...) 的结果保存到 O
hidden = zeros(T, numex);  % 详解: 赋值：将 zeros(...) 的结果保存到 hidden
obs = zeros(O, T, numex);  % 详解: 赋值：将 zeros(...) 的结果保存到 obs

hidden = mc_sample(initial_prob, transmat, T, numex)';  % 赋值：设置变量 hidden  % 详解: 赋值：将 mc_sample(...) 的结果保存到 hidden  % 详解: 赋值：将 mc_sample(...) 的结果保存到 hidden
for i=1:numex  % 详解: for 循环：迭代变量 i 遍历 1:numex
  for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
    q = hidden(t,i);  % 详解: 赋值：将 hidden(...) 的结果保存到 q
    m = sample_discrete(mixmat(q,:), 1, 1);  % 详解: 赋值：将 sample_discrete(...) 的结果保存到 m
    obs(:,t,i) =  gaussian_sample(mu(:,q,m), Sigma(:,:,q,m), 1);  % 详解: 调用函数：obs(:,t,i) = gaussian_sample(mu(:,q,m), Sigma(:,:,q,m), 1)
  end  % 详解: 执行语句
end  % 详解: 执行语句




