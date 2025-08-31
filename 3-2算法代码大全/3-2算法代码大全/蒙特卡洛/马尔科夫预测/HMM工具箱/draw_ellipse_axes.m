% 文件: draw_ellipse_axes.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function h = draw_ellipse_axes(x, c, linespec)  % 详解: 执行语句

[v,e] = eig(c);  % 详解: 执行语句
v = v*sqrt(e);  % 详解: 赋值：计算表达式并保存到 v

h = [];  % 详解: 赋值：计算表达式并保存到 h
for j = 1:cols(v)  % 详解: for 循环：迭代变量 j 遍历 1:cols(v)
  x1 = repmat(x(1,:),2,1) + repmat([-1;1]*v(1,j),1,cols(x));  % 详解: 赋值：将 repmat(...) 的结果保存到 x1
  x2 = repmat(x(2,:),2,1) + repmat([-1;1]*v(2,j),1,cols(x));  % 详解: 赋值：将 repmat(...) 的结果保存到 x2
  h = [h line(x1,x2)];  % 详解: 赋值：计算表达式并保存到 h
end  % 详解: 执行语句
if nargin > 2  % 详解: 条件判断：if (nargin > 2)
  set_linespec(h,linespec);  % 详解: 调用函数：set_linespec(h,linespec)
end  % 详解: 执行语句




