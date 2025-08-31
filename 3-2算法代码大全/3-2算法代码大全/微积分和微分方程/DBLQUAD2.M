% 文件: DBLQUAD2.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function I=dblquad2(f_name,a,b,c_lo,d_hi,m,n)  % 详解: 执行语句


if nargin<7, n=100;end  % 详解: 条件判断：if (nargin<7, n=100;end)
if nargin<6, m=100;end  % 详解: 条件判断：if (nargin<6, m=100;end)
if m<2|n<2  % 详解: 条件判断：if (m<2|n<2)
   error('Number of intervals invalid\n');  % 详解: 调用函数：error('Number of intervals invalid\n')
end  % 详解: 执行语句
mpt=m+1;hx=(b-a)/m;x=a+(0:m)*hx;  % 详解: 赋值：计算表达式并保存到 mpt
for i=1:mpt  % 详解: for 循环：迭代变量 i 遍历 1:mpt
  ylo=feval(c_lo,x(i));yhi=feval(d_hi,x(i));  % 详解: 赋值：将 feval(...) 的结果保存到 ylo
  hy=(yhi-ylo)/n; y(i,:)=ylo+(0:n)*hy;  % 详解: 赋值：计算表达式并保存到 hy
  f(i,:)=feval(f_name,x(i),y(i,:));  % 详解: 调用函数：f(i,:)=feval(f_name,x(i),y(i,:))
  G(i)=trapz(y(i,:),f(i,:));  % 详解: 调用函数：G(i)=trapz(y(i,:),f(i,:))
end  % 详解: 执行语句
I=trapz(x,G);  % 详解: 赋值：将 trapz(...) 的结果保存到 I




