% 文件: student_t_prob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = student_t_pdf(X, mu, lambda, alpha)  % 详解: 执行语句

k = length(mu);  % 详解: 赋值：将 length(...) 的结果保存到 k
assert(size(X,1) == k);  % 详解: 调用函数：assert(size(X,1) == k)
[k N] = size(X);  % 详解: 获取向量/矩阵尺寸
numer = gamma(0.5*(alpha+k));  % 详解: 赋值：将 gamma(...) 的结果保存到 numer
denom = gamma(0.5*alpha) * (alpha*pi)^(k/2);  % 详解: 赋值：将 gamma(...) 的结果保存到 denom
c = (numer/denom) * det(lambda)^(0.5);  % 详解: 赋值：计算表达式并保存到 c
p = c*(1 + (1/alpha)*(X-mu)'*lambda*(X-mu))^(-(alpha+k)/2); % scalar version  % 详解: 赋值：计算表达式并保存到 p  % 详解: 赋值：计算表达式并保存到 p

keyboard  % 详解: 执行语句




