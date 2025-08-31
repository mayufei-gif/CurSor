% 文件: pick.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [i,j] = pick(ndx)  % 详解: 函数定义：pick(ndx), 返回：i,j

dist = normalize(ones(1,length(ndx)));  % 详解: 赋值：将 normalize(...) 的结果保存到 dist
j = sample_discrete(dist);  % 详解: 赋值：将 sample_discrete(...) 的结果保存到 j
i = ndx(j);  % 详解: 赋值：将 ndx(...) 的结果保存到 i




