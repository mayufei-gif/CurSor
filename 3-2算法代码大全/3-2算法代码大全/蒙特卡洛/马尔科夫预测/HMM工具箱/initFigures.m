% 文件: initFigures.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% initFigures
% Position 6 figures on the edges of the screen.
% [xmin ymin w h] where (0,0) = bottom left
% Numbers assume screen resolution is 1024 x 1280

global FIGNUM NUMFIGS  % 详解: 执行语句
FIGNUM = 1; NUMFIGS = 6;  % 详解: 赋值：计算表达式并保存到 FIGNUM

screenMain = true;  % 详解: 赋值：计算表达式并保存到 screenMain

if screenMain  % 详解: 条件判断：if (screenMain)
  xoff = 0;  % 详解: 赋值：计算表达式并保存到 xoff
else  % 详解: 条件判断：else 分支
  xoff = -1280;  % 详解: 赋值：计算表达式并保存到 xoff
end  % 详解: 执行语句

w = 400; h = 300;  % 详解: 赋值：计算表达式并保存到 w
xs = [10 450 875] + xoff;  % 详解: 赋值：计算表达式并保存到 xs
ys = [650 40];  % 详解: 赋值：计算表达式并保存到 ys

if 0  % 详解: 条件判断：if (0)
w = 350; h = 250;  % 详解: 赋值：计算表达式并保存到 w
xs = [10 380 750]+xoff;  % 详解: 赋值：计算表达式并保存到 xs
ys = [700 350 10];  % 详解: 赋值：计算表达式并保存到 ys
end  % 详解: 执行语句


Nfigs = length(xs)*length(ys);  % 详解: 赋值：将 length(...) 的结果保存到 Nfigs
if screenMain  % 详解: 条件判断：if (screenMain)
  fig = 1;  % 详解: 赋值：计算表达式并保存到 fig
else  % 详解: 条件判断：else 分支
  fig = Nfigs + 1;  % 详解: 赋值：计算表达式并保存到 fig
end  % 详解: 执行语句

for yi=1:length(ys)  % 详解: for 循环：迭代变量 yi 遍历 1:length(ys)
  for xi=1:length(xs)  % 详解: for 循环：迭代变量 xi 遍历 1:length(xs)
    figure(fig);  % 详解: 调用函数：figure(fig)
    set(gcf, 'position', [xs(xi) ys(yi) w h]);  % 详解: 调用函数：set(gcf, 'position', [xs(xi) ys(yi) w h])
    fig = fig + 1;  % 详解: 赋值：计算表达式并保存到 fig
  end  % 详解: 执行语句
end  % 详解: 执行语句





