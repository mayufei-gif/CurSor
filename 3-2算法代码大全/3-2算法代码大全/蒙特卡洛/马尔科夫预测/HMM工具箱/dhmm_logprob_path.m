% 文件: dhmm_logprob_path.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [ll, p] = prob_path(prior, transmat, obsmat, qs)  % 详解: 函数定义：prob_path(prior, transmat, obsmat, qs), 返回：ll, p

T = size(obsmat, 2);  % 详解: 赋值：将 size(...) 的结果保存到 T
p = zeros(1,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 p
p(1) = prior(qs(1)) * obsmat(qs(1),1);  % 详解: 调用函数：p(1) = prior(qs(1)) * obsmat(qs(1),1)
for t=2:T  % 详解: for 循环：迭代变量 t 遍历 2:T
  p(t) = transmat(qs(t-1), qs(t)) * obsmat(qs(t),t);  % 详解: 调用函数：p(t) = transmat(qs(t-1), qs(t)) * obsmat(qs(t),t)
end  % 详解: 执行语句

ll = sum(log(p));  % 详解: 赋值：将 sum(...) 的结果保存到 ll




