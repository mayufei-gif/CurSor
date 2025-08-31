% 文件: RK4.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [tout, yout] = rk4(ypfun, tspan, y0, h)  % 详解: 函数定义：rk4(ypfun, tspan, y0, h), 返回：tout, yout


t=tspan(1):h:tspan(2);  % 详解: 赋值：将 tspan(...) 的结果保存到 t
y(:,1)=y0(:);  % 详解: 调用函数：y(:,1)=y0(:)

for i=1:length(t)-1,  % 详解: for 循环：迭代变量 i 遍历 1:length(t)-1,
   k1=feval(ypfun,t(i),y(:,i));  % 详解: 赋值：将 feval(...) 的结果保存到 k1
   k2=feval(ypfun,t(i)+h/2,y(:,i)+h*k1/2);  % 详解: 赋值：将 feval(...) 的结果保存到 k2
   k3=feval(ypfun,t(i)+h/2,y(:,i)+h*k2/2);  % 详解: 赋值：将 feval(...) 的结果保存到 k3
   k4=feval(ypfun,t(i)+h,y(:,i)+h*k3);  % 详解: 赋值：将 feval(...) 的结果保存到 k4
   y(:,i+1)=y(:,i)+h*(k1+2*k2+2*k3+k4)/6;  % 详解: 执行语句
end  % 详解: 执行语句

tout=t';yout=y';  % 详解: 赋值：计算表达式并保存到 tout



