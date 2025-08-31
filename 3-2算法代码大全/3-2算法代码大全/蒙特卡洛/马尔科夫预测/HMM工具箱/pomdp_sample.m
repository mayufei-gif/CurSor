% 文件: pomdp_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [obs, hidden] = pomdp_sample(initial_prob, transmat, obsmat, act)  % 详解: 函数定义：pomdp_sample(initial_prob, transmat, obsmat, act), 返回：obs, hidden


len = length(act);  % 详解: 赋值：将 length(...) 的结果保存到 len
hidden = mdp_sample(initial_prob, transmat, act);  % 详解: 赋值：将 mdp_sample(...) 的结果保存到 hidden
obs = zeros(1, len);  % 详解: 赋值：将 zeros(...) 的结果保存到 obs
for t=1:len  % 详解: for 循环：迭代变量 t 遍历 1:len
  obs(t) = sample_discrete(obsmat(hidden(t),:));  % 详解: 调用函数：obs(t) = sample_discrete(obsmat(hidden(t),:))
end  % 详解: 执行语句




