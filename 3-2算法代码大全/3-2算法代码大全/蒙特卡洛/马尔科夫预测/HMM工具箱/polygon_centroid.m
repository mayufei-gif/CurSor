% 文件: polygon_centroid.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x0,y0] = centroid(x,y)  % 详解: 函数定义：centroid(x,y), 返回：x0,y0



if nargin==0, help centroid, return, end  % 详解: 条件判断：if (nargin==0, help centroid, return, end)
if nargin==1  % 详解: 条件判断：if (nargin==1)
  sz = size(x);  % 详解: 赋值：将 size(...) 的结果保存到 sz
  if sz(1)==2  % 详解: 条件判断：if (sz(1)==2)
    y = x(2,:); x = x(1,:);  % 详解: 赋值：将 x(...) 的结果保存到 y
  elseif sz(2)==2  % 详解: 条件判断：elseif (sz(2)==2)
    y = x(:,2); x = x(:,1);  % 详解: 赋值：将 x(...) 的结果保存到 y
  else  % 详解: 条件判断：else 分支
    y = imag(x);  % 详解: 赋值：将 imag(...) 的结果保存到 y
    x = real(x);  % 详解: 赋值：将 real(...) 的结果保存到 x
  end  % 详解: 执行语句
end  % 详解: 执行语句

x = [x(:); x(1)];  % 详解: 赋值：计算表达式并保存到 x
y = [y(:); y(1)];  % 详解: 赋值：计算表达式并保存到 y

l = length(x);  % 详解: 赋值：将 length(...) 的结果保存到 l
if length(y)~=l  % 详解: 条件判断：if (length(y)~=l)
  error(' Vectors x and y must have the same length')  % 详解: 调用函数：error(' Vectors x and y must have the same length')
end  % 详解: 执行语句

del = y(2:l)-y(1:l-1);  % 详解: 赋值：将 y(...) 的结果保存到 del
v = x(1:l-1).^2+x(2:l).^2+x(1:l-1).*x(2:l);  % 详解: 赋值：将 x(...) 的结果保存到 v
x0 = v'*del;  % 赋值：设置变量 x0  % 详解: 赋值：计算表达式并保存到 x0  % 详解: 赋值：计算表达式并保存到 x0

del = x(2:l)-x(1:l-1);  % 详解: 赋值：将 x(...) 的结果保存到 del
v = y(1:l-1).^2+y(2:l).^2+y(1:l-1).*y(2:l);  % 详解: 赋值：将 y(...) 的结果保存到 v
y0 = v'*del;  % 赋值：设置变量 y0  % 详解: 赋值：计算表达式并保存到 y0  % 详解: 赋值：计算表达式并保存到 y0

a = (y(1:l-1)+y(2:l))'*del;  % 赋值：设置变量 a  % 详解: 赋值：计算表达式并保存到 a  % 详解: 赋值：计算表达式并保存到 a
tol= 2*eps;  % 详解: 赋值：计算表达式并保存到 tol
if abs(a)<tol  % 详解: 条件判断：if (abs(a)<tol)
  disp(' Warning: area of polygon is close to 0')  % 详解: 调用函数：disp(' Warning: area of polygon is close to 0')
  a = a+sign(a)*tol+(~a)*tol;  % 详解: 赋值：计算表达式并保存到 a
end  % 详解: 执行语句
a = 1/3/a;  % 详解: 赋值：计算表达式并保存到 a

x0 = -x0*a;  % 详解: 赋值：计算表达式并保存到 x0
y0 =  y0*a;  % 详解: 赋值：计算表达式并保存到 y0

if nargout < 2, x0 = x0+i*y0; end  % 详解: 条件判断：if (nargout < 2, x0 = x0+i*y0; end)




