% 文件: matrix_normal_pdf.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = matrix_normal_pdf(A, M, V, K)  % 详解: 执行语句


[d m] = size(K);  % 详解: 获取向量/矩阵尺寸
c = det(K)^(d/2) / det(2*pi*V)^(m/2);  % 详解: 赋值：将 det(...) 的结果保存到 c
p = c * exp(-0.5*tr((A-M)'*inv(V)*(A-M)*K));  % 矩阵求逆  % 详解: 赋值：计算表达式并保存到 p  % 详解: 赋值：计算表达式并保存到 p




