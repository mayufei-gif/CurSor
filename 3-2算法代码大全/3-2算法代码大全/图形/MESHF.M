% 文件: MESHF.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x,y,z]=meshf(f_name,xa,xb,y_lower,y_upper,m,n)  % 详解: 函数定义：meshf(f_name,xa,xb,y_lower,y_upper,m,n), 返回：x,y,z


if nargin<7,n=20;end  % 详解: 条件判断：if (nargin<7,n=20;end)
if nargin<6,m=20;end  % 详解: 条件判断：if (nargin<6,m=20;end)

x=linspace(xa,xb,m);  % 详解: 赋值：将 linspace(...) 的结果保存到 x
yc=feval(y_lower,x);yd=feval(y_upper,x);  % 详解: 赋值：将 feval(...) 的结果保存到 yc

x=ones(n,1)*x;  % 详解: 赋值：将 ones(...) 的结果保存到 x
y=zeros(n,m);  % 详解: 赋值：将 zeros(...) 的结果保存到 y
for i=1:m  % 详解: for 循环：迭代变量 i 遍历 1:m
  y(:,i)=linspace(yc(i),yd(i),n)';  % 生成等/对数间隔序列  % 详解: 执行语句  % 详解: 执行语句
end  % 详解: 执行语句
z=feval(f_name,x,y);  % 详解: 赋值：将 feval(...) 的结果保存到 z

if nargout==0,mesh(x,y,z);xlabel('X');ylabel('Y');end  % 详解: 条件判断：if (nargout==0,mesh(x,y,z);xlabel('X');ylabel('Y');end)




