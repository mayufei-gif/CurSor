% 文件: beta_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function r = betarnd(a,b,m,n);  % 详解: 执行语句



if nargin < 2,  % 详解: 条件判断：if (nargin < 2,)
    error('Requires at least two input arguments');  % 详解: 调用函数：error('Requires at least two input arguments')
end  % 详解: 执行语句

if nargin == 2  % 详解: 条件判断：if (nargin == 2)
    [errorcode rows columns] = rndcheck(2,2,a,b);  % 详解: 执行语句
end  % 详解: 执行语句

if nargin == 3  % 详解: 条件判断：if (nargin == 3)
    [errorcode rows columns] = rndcheck(3,2,a,b,m);  % 详解: 执行语句
end  % 详解: 执行语句

if nargin == 4  % 详解: 条件判断：if (nargin == 4)
    [errorcode rows columns] = rndcheck(4,2,a,b,m,n);  % 详解: 执行语句
end  % 详解: 执行语句

if errorcode > 0  % 详解: 条件判断：if (errorcode > 0)
    error('Size information is inconsistent.');  % 详解: 调用函数：error('Size information is inconsistent.')
end  % 详解: 执行语句

r = zeros(rows,columns);  % 详解: 赋值：将 zeros(...) 的结果保存到 r

if prod(size(a)) == 1  % 详解: 条件判断：if (prod(size(a)) == 1)
    a1 = a(ones(rows,1),ones(columns,1));  % 详解: 赋值：将 a(...) 的结果保存到 a1
    g1 = gamrnd(a1,1);  % 详解: 赋值：将 gamrnd(...) 的结果保存到 g1
else  % 详解: 条件判断：else 分支
    g1 = gamrnd(a,1);  % 详解: 赋值：将 gamrnd(...) 的结果保存到 g1
end  % 详解: 执行语句
if prod(size(b)) == 1  % 详解: 条件判断：if (prod(size(b)) == 1)
    b1 = b(ones(rows,1),ones(columns,1));  % 详解: 赋值：将 b(...) 的结果保存到 b1
    g2 = gamrnd(b1,1);  % 详解: 赋值：将 gamrnd(...) 的结果保存到 g2
else  % 详解: 条件判断：else 分支
    g2 = gamrnd(b,1);  % 详解: 赋值：将 gamrnd(...) 的结果保存到 g2
end  % 详解: 执行语句
r = g1 ./ (g1 + g2);  % 详解: 赋值：计算表达式并保存到 r

if any(any(b <= 0));  % 详解: 条件判断：if (any(any(b <= 0));)
    if prod(size(b) == 1)  % 详解: 条件判断：if (prod(size(b) == 1))
        tmp = NaN;  % 详解: 赋值：计算表达式并保存到 tmp
        r = tmp(ones(rows,columns));  % 详解: 赋值：将 tmp(...) 的结果保存到 r
    else  % 详解: 条件判断：else 分支
        k = find(b <= 0);  % 详解: 赋值：将 find(...) 的结果保存到 k
        tmp = NaN;  % 详解: 赋值：计算表达式并保存到 tmp
        r(k) = tmp(ones(size(k)));  % 详解: 调用函数：r(k) = tmp(ones(size(k)))
    end  % 详解: 执行语句
end  % 详解: 执行语句

if any(any(a <= 0));  % 详解: 条件判断：if (any(any(a <= 0));)
    if prod(size(a) == 1)  % 详解: 条件判断：if (prod(size(a) == 1))
        tmp = NaN;  % 详解: 赋值：计算表达式并保存到 tmp
        r = tmp(ones(rows,columns));  % 详解: 赋值：将 tmp(...) 的结果保存到 r
    else  % 详解: 条件判断：else 分支
        k = find(a <= 0);  % 详解: 赋值：将 find(...) 的结果保存到 k
        tmp = NaN;  % 详解: 赋值：计算表达式并保存到 tmp
        r(k) = tmp(ones(size(k)));  % 详解: 调用函数：r(k) = tmp(ones(size(k)))
    end  % 详解: 执行语句
end  % 详解: 执行语句




