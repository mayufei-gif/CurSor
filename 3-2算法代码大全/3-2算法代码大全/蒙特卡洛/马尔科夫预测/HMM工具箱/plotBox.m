% 文件: plotBox.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [h, ht] =plotBox(box, col, str)  % 详解: 函数定义：plotBox(box, col, str), 返回：h, ht

if nargin < 2, col = 'r'; end  % 详解: 条件判断：if (nargin < 2, col = 'r'; end)
if nargin < 3, str = ''; end  % 详解: 条件判断：if (nargin < 3, str = ''; end)

box = double(box);  % 详解: 赋值：将 double(...) 的结果保存到 box

h = plot([box(1) box(2) box(2) box(1) box(1)], [ box(3) box(3) box(4) box(4) box(3)]);  % 详解: 赋值：将 plot(...) 的结果保存到 h
set(h, 'color', col);  % 详解: 调用函数：set(h, 'color', col)
set(h, 'linewidth', 2);  % 详解: 调用函数：set(h, 'linewidth', 2)
if ~isempty(str)  % 详解: 条件判断：if (~isempty(str))
  xc = mean(box(1:2));  % 详解: 赋值：将 mean(...) 的结果保存到 xc
  yc = mean(box(3:4));  % 详解: 赋值：将 mean(...) 的结果保存到 yc
  ht = text(xc, yc, str);  % 详解: 赋值：将 text(...) 的结果保存到 ht
else  % 详解: 条件判断：else 分支
  ht = [];  % 详解: 赋值：计算表达式并保存到 ht
end  % 详解: 执行语句




