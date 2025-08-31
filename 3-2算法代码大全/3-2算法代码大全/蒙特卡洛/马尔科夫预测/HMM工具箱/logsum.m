% 文件: logsum.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function y=logsum(x,d)  % 详解: 执行语句


if nargin==1  % 详解: 条件判断：if (nargin==1)
   d=[find(size(x)-1) 1];  % 详解: 赋值：计算表达式并保存到 d
   d=d(1);  % 详解: 赋值：将 d(...) 的结果保存到 d
end  % 详解: 执行语句
n=size(x,d);  % 详解: 赋值：将 size(...) 的结果保存到 n
if n<=1, y=x; return; end  % 详解: 条件判断：if (n<=1, y=x; return; end)
s=size(x);  % 详解: 赋值：将 size(...) 的结果保存到 s
p=[d:ndims(x) 1:d-1];  % 详解: 赋值：计算表达式并保存到 p
z=reshape(permute(x,p),n,prod(s)/n);  % 详解: 赋值：将 reshape(...) 的结果保存到 z

y=max(z);  % 详解: 赋值：将 max(...) 的结果保存到 y
y=y+log(sum(exp(z-y(ones(n,1),:))));  % 详解: 赋值：计算表达式并保存到 y

s(d)=1;  % 详解: 执行语句
y=ipermute(reshape(y,s(p)),p);  % 详解: 赋值：将 ipermute(...) 的结果保存到 y





