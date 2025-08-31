% 文件: NEWTON.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function x=newton(f_name,x0,tol)  % 详解: 执行语句


h=0.0001;M=500;  % 详解: 赋值：计算表达式并保存到 h
if nargin<3, tol=1e-6;end  % 详解: 条件判断：if (nargin<3, tol=1e-6;end)
x=x0;xb=x-999;  % 详解: 赋值：计算表达式并保存到 x
n=0;  % 详解: 赋值：计算表达式并保存到 n
while abs(x-xb)>tol  % 详解: while 循环：当 (abs(x-xb)>tol) 为真时迭代
  xb=x;  % 详解: 赋值：计算表达式并保存到 xb
  if n>M break;end  % 详解: 条件判断：if (n>M break;end)
  y=feval(f_name, x);  % 详解: 赋值：将 feval(...) 的结果保存到 y
  fprintf('  n=%3.0f,  x=%12.5e,  y=%12.5e, \n',n,x,y)  % 详解: 调用函数：fprintf(' n=%3.0f, x=%12.5e, y=%12.5e, \n',n,x,y)
  y_driv=(feval(f_name,x+h)-y)/h;  % 详解: 赋值：计算表达式并保存到 y_driv
  x=xb-y/y_driv;  % 详解: 赋值：计算表达式并保存到 x
  n=n+1;  % 详解: 赋值：计算表达式并保存到 n
end  % 详解: 执行语句
fprintf('  n=%3.0f,  x=%12.5e,  y=%12.5e, ',n,x,y)  % 详解: 调用函数：fprintf(' n=%3.0f, x=%12.5e, y=%12.5e, ',n,x,y)
if n>500,  % 详解: 条件判断：if (n>500,)
 fprintf('\n');disp('Warning: iterations exceeds the limitation, probable devergent');  % 详解: 调用函数：fprintf('\n');disp('Warning: iterations exceeds the limitation, probable devergent')
end  % 详解: 执行语句



