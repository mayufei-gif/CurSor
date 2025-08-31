% 文件: normal_coef.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function c = normal_coef (Sigma)  % 详解: 执行语句

n = length(Sigma);  % 详解: 赋值：将 length(...) 的结果保存到 n
c = (2*pi)^(-n/2) * det(Sigma)^(-0.5);  % 详解: 赋值：计算表达式并保存到 c





