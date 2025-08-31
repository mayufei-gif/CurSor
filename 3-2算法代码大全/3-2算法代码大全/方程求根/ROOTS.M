% 文件: ROOTS.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function r = roots(c)  % 详解: 执行语句



if size(c,1)>1 & size(c,2)>1  % 详解: 条件判断：if (size(c,1)>1 & size(c,2)>1)
    error('Must be a vector.')  % 详解: 调用函数：error('Must be a vector.')
end  % 详解: 执行语句
c = c(:).';  % 赋值：设置变量 c  % 详解: 赋值：将 c(...) 的结果保存到 c  % 详解: 赋值：将 c(...) 的结果保存到 c
n = size(c,2);  % 详解: 赋值：将 size(...) 的结果保存到 n
r = zeros(0,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 r

inz = find(c);  % 详解: 赋值：将 find(...) 的结果保存到 inz
if isempty(inz),  % 详解: 条件判断：if (isempty(inz),)
    return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

nnz = length(inz);  % 详解: 赋值：将 length(...) 的结果保存到 nnz
c = c(inz(1):inz(nnz));  % 详解: 赋值：将 c(...) 的结果保存到 c
r = zeros(n-inz(nnz),1);  % 详解: 赋值：将 zeros(...) 的结果保存到 r

n = length(c);  % 详解: 赋值：将 length(...) 的结果保存到 n
if n > 1  % 详解: 条件判断：if (n > 1)
    a = diag(ones(1,n-2),-1);  % 详解: 赋值：将 diag(...) 的结果保存到 a
    a(1,:) = -c(2:n) ./ c(1);  % 详解: 调用函数：a(1,:) = -c(2:n) ./ c(1)
    r = [r;eig(a)];  % 详解: 赋值：计算表达式并保存到 r
end  % 详解: 执行语句




