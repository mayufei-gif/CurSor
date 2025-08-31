% 文件: multirnd.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function r = multirnd(theta,k)  % 详解: 执行语句

error(nargchk(1,2,nargin));  % 详解: 调用函数：error(nargchk(1,2,nargin))

if ndims(theta) > 2 | all(size(theta) > 1)  % 详解: 条件判断：if (ndims(theta) > 2 | all(size(theta) > 1))
    error('theta must be a vector');  % 详解: 调用函数：error('theta must be a vector')
end  % 详解: 执行语句

if size(theta,1) == 1  % 详解: 条件判断：if (size(theta,1) == 1)
    theta = theta';  % 赋值：设置变量 theta  % 详解: 赋值：计算表达式并保存到 theta  % 详解: 赋值：计算表达式并保存到 theta
end  % 详解: 执行语句


if nargin == 1  % 详解: 条件判断：if (nargin == 1)
    k = 1;  % 详解: 赋值：计算表达式并保存到 k
end  % 详解: 执行语句


n = length(theta);  % 详解: 赋值：将 length(...) 的结果保存到 n
theta_cdf = cumsum(theta);  % 详解: 赋值：将 cumsum(...) 的结果保存到 theta_cdf

r = zeros(n,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 r
random_vals = rand(k,1);  % 详解: 赋值：将 rand(...) 的结果保存到 random_vals

for j = 1:k  % 详解: for 循环：迭代变量 j 遍历 1:k
    index = min(find(random_vals(j) <= theta_cdf));  % 详解: 赋值：将 min(...) 的结果保存到 index
    r(index) = r(index) + 1;  % 详解: 执行语句
end  % 详解: 执行语句



