% 文件: dhmm_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [obs, hidden] = dhmm_sample(initial_prob, transmat, obsmat, numex, len)  % 详解: 函数定义：dhmm_sample(initial_prob, transmat, obsmat, numex, len), 返回：obs, hidden

hidden = mc_sample(initial_prob, transmat, len, numex);  % 详解: 赋值：将 mc_sample(...) 的结果保存到 hidden
obs = multinomial_sample(hidden, obsmat);  % 详解: 赋值：将 multinomial_sample(...) 的结果保存到 obs






