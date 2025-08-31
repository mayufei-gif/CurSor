% 文件: axis_pct.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function ax = axis_pct(pct)  % 详解: 执行语句

if nargin < 1  % 详解: 条件判断：if (nargin < 1)
  pct = 0.05;  % 详解: 赋值：计算表达式并保存到 pct
end  % 详解: 执行语句
ax = [Inf -Inf Inf -Inf Inf -Inf];  % 详解: 赋值：计算表达式并保存到 ax

children = get(gca,'children');  % 详解: 赋值：将 get(...) 的结果保存到 children
for child = children'  % for 循环：遍历迭代  % 详解: for 循环：迭代变量 child 遍历 children'  % 详解: for 循环：迭代变量 child 遍历 children'  % for 循环：遍历迭代  % 详解: for 循环：迭代变量 child 遍历 children'
  if strcmp(get(child,'type'),'text')  % 详解: 条件判断：if (strcmp(get(child,'type'),'text'))
    xyz = get(child,'position');  % 详解: 赋值：将 get(...) 的结果保存到 xyz
    c([1 2]) = xyz(1);  % 详解: 调用函数：c([1 2]) = xyz(1)
    c([3 4]) = xyz(2);  % 详解: 调用函数：c([3 4]) = xyz(2)
    c([5 6]) = xyz(3);  % 详解: 调用函数：c([5 6]) = xyz(3)
  else  % 详解: 条件判断：else 分支
    x = get(child,'xdata');  % 详解: 赋值：将 get(...) 的结果保存到 x
    c(1) = min(x);  % 详解: 调用函数：c(1) = min(x)
    c(2) = max(x);  % 详解: 调用函数：c(2) = max(x)
    y = get(child,'ydata');  % 详解: 赋值：将 get(...) 的结果保存到 y
    c(3) = min(y);  % 详解: 调用函数：c(3) = min(y)
    c(4) = max(y);  % 详解: 调用函数：c(4) = max(y)
    z = get(child,'zdata');  % 详解: 赋值：将 get(...) 的结果保存到 z
    if isempty(z)  % 详解: 条件判断：if (isempty(z))
      c([5 6]) = 0;  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
      c(5) = min(z);  % 详解: 调用函数：c(5) = min(z)
      c(6) = max(z);  % 详解: 调用函数：c(6) = max(z)
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  ax([1 3 5]) = min(ax([1 3 5]), c([1 3 5]));  % 详解: 调用函数：ax([1 3 5]) = min(ax([1 3 5]), c([1 3 5]))
  ax([2 4 6]) = max(ax([2 4 6]), c([2 4 6]));  % 详解: 调用函数：ax([2 4 6]) = max(ax([2 4 6]), c([2 4 6]))
end  % 详解: 执行语句
if strcmp(get(gca,'xscale'), 'log')  % 详解: 条件判断：if (strcmp(get(gca,'xscale'), 'log'))
  ax([1 2]) = log(ax([1 2]));  % 详解: 调用函数：ax([1 2]) = log(ax([1 2]))
end  % 详解: 执行语句
if strcmp(get(gca,'yscale'), 'log')  % 详解: 条件判断：if (strcmp(get(gca,'yscale'), 'log'))
  ax([3 4]) = log(ax([3 4]));  % 详解: 调用函数：ax([3 4]) = log(ax([3 4]))
end  % 详解: 执行语句
dx = ax(2)-ax(1);  % 详解: 赋值：将 ax(...) 的结果保存到 dx
if dx == 0  % 详解: 条件判断：if (dx == 0)
  dx = 1;  % 详解: 赋值：计算表达式并保存到 dx
end  % 详解: 执行语句
dy = ax(4)-ax(3);  % 详解: 赋值：将 ax(...) 的结果保存到 dy
if dy == 0  % 详解: 条件判断：if (dy == 0)
  dy = 1;  % 详解: 赋值：计算表达式并保存到 dy
end  % 详解: 执行语句
dz = ax(6)-ax(5);  % 详解: 赋值：将 ax(...) 的结果保存到 dz
if dz == 0  % 详解: 条件判断：if (dz == 0)
  dz = 1;  % 详解: 赋值：计算表达式并保存到 dz
end  % 详解: 执行语句
ax = ax + [-dx dx -dy dy -dz dz]*pct;  % 详解: 赋值：计算表达式并保存到 ax
if strcmp(get(gca,'xscale'), 'log')  % 详解: 条件判断：if (strcmp(get(gca,'xscale'), 'log'))
  ax([1 2]) = exp(ax([1 2]));  % 详解: 调用函数：ax([1 2]) = exp(ax([1 2]))
end  % 详解: 执行语句
if strcmp(get(gca,'yscale'), 'log')  % 详解: 条件判断：if (strcmp(get(gca,'yscale'), 'log'))
  ax([3 4]) = exp(ax([3 4]));  % 详解: 调用函数：ax([3 4]) = exp(ax([3 4]))
end  % 详解: 执行语句
ax = ax(1:length(axis));  % 详解: 赋值：将 ax(...) 的结果保存到 ax
axis(ax);  % 详解: 调用函数：axis(ax)
if nargout < 1  % 详解: 条件判断：if (nargout < 1)
  clear ax  % 详解: 执行语句
end  % 详解: 执行语句




