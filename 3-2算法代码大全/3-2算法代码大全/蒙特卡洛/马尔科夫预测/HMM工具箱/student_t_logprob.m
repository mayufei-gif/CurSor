% 文件: student_t_logprob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function L = log_student_pdf(X, mu, lambda, alpha)  % 详解: 执行语句

k = length(mu);  % 详解: 赋值：将 length(...) 的结果保存到 k
assert(size(X,1) == k);  % 详解: 调用函数：assert(size(X,1) == k)
[k N] = size(X);  % 详解: 获取向量/矩阵尺寸
logc = gammaln(0.5*(alpha+k)) - gammaln(0.5*alpha) - (k/2)*log(alpha*pi) + 0.5*log(det(lambda));  % 详解: 赋值：将 gammaln(...) 的结果保存到 logc
middle = (1 + (1/alpha)*(X-mu)'*lambda*(X-mu)); % scalar version  % 详解: 赋值：计算表达式并保存到 middle  % 详解: 赋值：计算表达式并保存到 middle
L = logc - ((alpha+k)/2)*log(middle);  % 详解: 赋值：计算表达式并保存到 L




