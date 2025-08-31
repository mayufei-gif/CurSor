% 文件: dirichletpdf.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = dirichletpdf(x, alpha)  % 详解: 执行语句

error(nargchk(2,2,nargin));  % 详解: 调用函数：error(nargchk(2,2,nargin))

if min(size(alpha)) ~= 1 | ndims(alpha) > 2 | length(alpha) == 1  % 详解: 条件判断：if (min(size(alpha)) ~= 1 | ndims(alpha) > 2 | length(alpha) == 1)
    error('alpha must be a vector');  % 详解: 调用函数：error('alpha must be a vector')
end  % 详解: 执行语句

if any(size(x) ~= size(alpha))  % 详解: 条件判断：if (any(size(x) ~= size(alpha)))
    error('x and alpha must be the same size');  % 详解: 调用函数：error('x and alpha must be the same size')
end  % 详解: 执行语句


if any(x < 0)  % 详解: 条件判断：if (any(x < 0))
    p = 0;  % 详解: 赋值：计算表达式并保存到 p
elseif sum(x) ~= 1  % 详解: 条件判断：elseif (sum(x) ~= 1)
    disp(['dirichletpdf warning: sum(x)~=1, but this may be ' ...  % 详解: 统计：求和/均值/中位数
        'due to numerical issues']);  % 详解: 执行语句
    p = 0;  % 详解: 赋值：计算表达式并保存到 p
else  % 详解: 条件判断：else 分支
    z = gammaln(sum(alpha)) - sum(gammaln(alpha));  % 详解: 赋值：将 gammaln(...) 的结果保存到 z
    z = exp(z);  % 详解: 赋值：将 exp(...) 的结果保存到 z

    p = z * prod(x.^(alpha-1));  % 详解: 赋值：计算表达式并保存到 p
end  % 详解: 执行语句



