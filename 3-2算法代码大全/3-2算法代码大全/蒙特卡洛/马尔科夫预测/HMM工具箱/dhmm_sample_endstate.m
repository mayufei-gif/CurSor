% 文件: dhmm_sample_endstate.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [obs, hidden] = dhmm_sample_endstate(startprob, transmat, obsmat, endprob, numex)  % 详解: 函数定义：dhmm_sample_endstate(startprob, transmat, obsmat, endprob, numex), 返回：obs, hidden

hidden = cell(1,numex);  % 详解: 赋值：将 cell(...) 的结果保存到 hidden
obs = cell(1,numex);  % 详解: 赋值：将 cell(...) 的结果保存到 obs

for m=1:numex  % 详解: for 循环：迭代变量 m 遍历 1:numex
  hidden{m} = mc_sample_endstate(startprob, transmat, endprob);  % 详解: 执行语句
  T = length(hidden{m});  % 详解: 赋值：将 length(...) 的结果保存到 T
  obs{m} = zeros(1,T);  % 详解: 创建零矩阵/数组
  for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
    h = hidden{m}(t);  % 详解: 赋值：计算表达式并保存到 h
    obs{m}(t) = sample_discrete(obsmat(h,:));  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句




