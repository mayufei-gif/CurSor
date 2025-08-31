% 文件: centeringMatrix.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

N = 3;  % 详解: 赋值：计算表达式并保存到 N
x = rand(N,2);  % 详解: 赋值：将 rand(...) 的结果保存到 x
m = mean(x,1);  % 详解: 赋值：将 mean(...) 的结果保存到 m
xc = x-repmat(m, N, 1);  % 详解: 赋值：计算表达式并保存到 xc

C = eye(N) - (1/N)*ones(N,N);  % 详解: 赋值：将 eye(...) 的结果保存到 C
xc2 = C*x;  % 详解: 赋值：计算表达式并保存到 xc2
assert(approxeq(xc, xc2))  % 详解: 调用函数：assert(approxeq(xc, xc2))




