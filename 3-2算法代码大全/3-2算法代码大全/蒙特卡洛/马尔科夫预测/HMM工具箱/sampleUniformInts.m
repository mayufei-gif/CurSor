% 文件: sampleUniformInts.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function M = sampleUniformInts(N, r, c)  % 详解: 执行语句


prob = normalize(ones(N,1));  % 详解: 赋值：将 normalize(...) 的结果保存到 prob
M = sample_discrete(prob, r, c);  % 详解: 赋值：将 sample_discrete(...) 的结果保存到 M




