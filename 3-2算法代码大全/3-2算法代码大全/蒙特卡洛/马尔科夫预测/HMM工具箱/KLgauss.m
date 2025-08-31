% 文件: KLgauss.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function kl = KLgauss(P, Q)  % 详解: 执行语句

R = P*inv(Q);  % 详解: 赋值：计算表达式并保存到 R
kl = -0.5*(log(det(R))) + trace(eye(length(P))-R);  % 详解: 赋值：计算表达式并保存到 kl





