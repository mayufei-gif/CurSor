% 文件: ind2subv.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function sub = ind2subv(siz, ndx)  % 详解: 执行语句

n = length(siz);  % 详解: 赋值：将 length(...) 的结果保存到 n

if n==0  % 详解: 条件判断：if (n==0)
  sub = ndx;  % 详解: 赋值：计算表达式并保存到 sub
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

if all(siz==2)  % 详解: 条件判断：if (all(siz==2))
  sub = dec2bitv(ndx-1, n);  % 详解: 赋值：将 dec2bitv(...) 的结果保存到 sub
  sub = sub(:,n:-1:1)+1;  % 详解: 赋值：将 sub(...) 的结果保存到 sub
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

cp = [1 cumprod(siz(:)')];  % 赋值：设置变量 cp  % 详解: 赋值：计算表达式并保存到 cp  % 详解: 赋值：计算表达式并保存到 cp
ndx = ndx(:) - 1;  % 详解: 赋值：将 ndx(...) 的结果保存到 ndx
sub = zeros(length(ndx), n);  % 详解: 赋值：将 zeros(...) 的结果保存到 sub
for i = n:-1:1  % 详解: for 循环：迭代变量 i 遍历 n:-1:1
  sub(:,i) = floor(ndx/cp(i))+1;  % 详解: 执行语句
  ndx = rem(ndx,cp(i));  % 详解: 赋值：将 rem(...) 的结果保存到 ndx
end  % 详解: 执行语句


function bits = dec2bitv(d,n)  % 详解: 执行语句


if (nargin<2)  % 详解: 条件判断：if ((nargin<2))
  n=1;  % 详解: 赋值：计算表达式并保存到 n
end  % 详解: 执行语句
d = d(:);  % 详解: 赋值：将 d(...) 的结果保存到 d

[f,e]=log2(max(d));  % 详解: 统计：最大/最小值
bits=rem(floor(d*pow2(1-max(n,e):0)),2);  % 详解: 赋值：将 rem(...) 的结果保存到 bits




