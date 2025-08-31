% 文件: draw_ellipse.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function h = draw_ellipse(x, c, outline_color, fill_color)  % 详解: 执行语句

n = 40;  % 详解: 赋值：计算表达式并保存到 n
radians = [0:(2*pi)/(n-1):2*pi];  % 详解: 赋值：计算表达式并保存到 radians
unitC = [sin(radians); cos(radians)];  % 详解: 赋值：计算表达式并保存到 unitC
r = chol(c)';  % 赋值：设置变量 r  % 详解: 赋值：将 chol(...) 的结果保存到 r  % 详解: 赋值：将 chol(...) 的结果保存到 r

if nargin < 3  % 详解: 条件判断：if (nargin < 3)
  outline_color = 'g';  % 详解: 赋值：计算表达式并保存到 outline_color
end  % 详解: 执行语句

h = [];  % 详解: 赋值：计算表达式并保存到 h
for i=1:cols(x)  % 详解: for 循环：迭代变量 i 遍历 1:cols(x)
  y = r*unitC + repmat(x(:, i), 1, n);  % 详解: 赋值：计算表达式并保存到 y
  if nargin < 4  % 详解: 条件判断：if (nargin < 4)
    h = [h line(y(1,:), y(2,:), 'Color', outline_color)];  % 详解: 赋值：计算表达式并保存到 h
  else  % 详解: 条件判断：else 分支
    h = [h fill(y(1,:), y(2,:), fill_color, 'EdgeColor', outline_color)];  % 详解: 赋值：计算表达式并保存到 h
  end  % 详解: 执行语句
end  % 详解: 执行语句




