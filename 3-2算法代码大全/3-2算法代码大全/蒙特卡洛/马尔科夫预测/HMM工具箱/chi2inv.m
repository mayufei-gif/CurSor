% 文件: chi2inv.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function x = chi2inv(p,v);  % 详解: 执行语句



if nargin < 2,  % 详解: 条件判断：if (nargin < 2,)
    error('Requires two input arguments.');  % 详解: 调用函数：error('Requires two input arguments.')
end  % 详解: 执行语句

[errorcode p v] = distchck(2,p,v);  % 详解: 执行语句

if errorcode > 0  % 详解: 条件判断：if (errorcode > 0)
    error('Requires non-scalar arguments to match in size.');  % 详解: 调用函数：error('Requires non-scalar arguments to match in size.')
end  % 详解: 执行语句

x = gaminv(p,v/2,2);  % 详解: 赋值：将 gaminv(...) 的结果保存到 x

k = (v <= 0);  % 详解: 赋值：计算表达式并保存到 k
if any(k(:))  % 详解: 条件判断：if (any(k(:)))
    x(k) = NaN;  % 详解: 执行语句
end  % 详解: 执行语句




