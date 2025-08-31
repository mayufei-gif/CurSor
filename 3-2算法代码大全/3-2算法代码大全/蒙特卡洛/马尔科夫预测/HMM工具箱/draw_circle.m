% 文件: draw_circle.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function h = draw_circle(x, r, outline_color, fill_color)  % 详解: 执行语句

n = 40;  % 详解: 赋值：计算表达式并保存到 n
radians = [0:(2*pi)/(n-1):2*pi];  % 详解: 赋值：计算表达式并保存到 radians
unitC = [sin(radians); cos(radians)];  % 详解: 赋值：计算表达式并保存到 unitC

if length(r) < cols(x)  % 详解: 条件判断：if (length(r) < cols(x))
  r = [r repmat(r(length(r)), 1, cols(x)-length(r))];  % 详解: 赋值：计算表达式并保存到 r
end  % 详解: 执行语句

h = [];  % 详解: 赋值：计算表达式并保存到 h
held = ishold;  % 详解: 赋值：计算表达式并保存到 held
hold on  % 详解: 执行语句
for i=1:cols(x)  % 详解: for 循环：迭代变量 i 遍历 1:cols(x)
  y = unitC*r(i) + repmat(x(:, i), 1, n);  % 详解: 赋值：计算表达式并保存到 y
  if nargin < 4  % 详解: 条件判断：if (nargin < 4)
    h = [h line(y(1,:), y(2,:), 'Color', outline_color)];  % 详解: 赋值：计算表达式并保存到 h
  else  % 详解: 条件判断：else 分支
    h = [h fill(y(1,:), y(2,:), fill_color, 'EdgeColor', outline_color)];  % 详解: 赋值：计算表达式并保存到 h
  end  % 详解: 执行语句
end  % 详解: 执行语句
if ~held  % 详解: 条件判断：if (~held)
  hold off  % 详解: 执行语句
end  % 详解: 执行语句




