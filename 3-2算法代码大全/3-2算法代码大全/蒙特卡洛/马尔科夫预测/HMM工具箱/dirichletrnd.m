% 文件: dirichletrnd.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function x = dirichletrnd(alpha)  % 详解: 执行语句

error(nargchk(1,1,nargin));  % 详解: 调用函数：error(nargchk(1,1,nargin))

if min(size(alpha)) ~= 1 | length(alpha) < 2  % 详解: 条件判断：if (min(size(alpha)) ~= 1 | length(alpha) < 2)
    error('alpha must be a vector of length at least 2');  % 详解: 调用函数：error('alpha must be a vector of length at least 2')
end  % 详解: 执行语句


gamma_vals = gamrnd(alpha, ones(size(alpha)), size(alpha));  % 详解: 赋值：将 gamrnd(...) 的结果保存到 gamma_vals
denom = sum(gamma_vals);  % 详解: 赋值：将 sum(...) 的结果保存到 denom
x = gamma_vals / denom;  % 详解: 赋值：计算表达式并保存到 x



