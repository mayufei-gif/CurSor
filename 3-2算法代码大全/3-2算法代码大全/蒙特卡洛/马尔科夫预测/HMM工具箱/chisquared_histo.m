% 文件: chisquared_histo.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function s = chisquared_histo(h1, h2)  % 详解: 执行语句
denom = h1 + h2;  % 详解: 赋值：计算表达式并保存到 denom
denom = denom + (denom==0);  % 详解: 赋值：计算表达式并保存到 denom
s = sum(((h1 - h2) .^ 2) ./ denom);  % 详解: 赋值：将 sum(...) 的结果保存到 s




