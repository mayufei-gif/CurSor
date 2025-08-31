% 文件: set_xtick_label.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function set_xtick_label(tick_labels, angle, axis_label)  % 详解: 函数定义：set_xtick_label(tick_labels, angle, axis_label)

if nargin < 2, angle = 90; end  % 详解: 条件判断：if (nargin < 2, angle = 90; end)
if nargin < 3, axis_label = []; end  % 详解: 条件判断：if (nargin < 3, axis_label = []; end)

pos = get(gca,'Position');  % 详解: 赋值：将 get(...) 的结果保存到 pos

ax = axis;  % 详解: 赋值：计算表达式并保存到 ax
axis(axis);  % 详解: 调用函数：axis(axis)
Yl = ax(3:4);  % 详解: 赋值：将 ax(...) 的结果保存到 Yl

set(gca, 'xtick', 0.7:1:length(tick_labels));  % 详解: 调用函数：set(gca, 'xtick', 0.7:1:length(tick_labels))
Xt = get(gca, 'xtick');  % 详解: 赋值：将 get(...) 的结果保存到 Xt

t = text(Xt,Yl(1)*ones(1,length(Xt)),tick_labels);  % 详解: 赋值：将 text(...) 的结果保存到 t
set(t,'HorizontalAlignment','right','VerticalAlignment','top', 'Rotation', angle);  % 详解: 调用函数：set(t,'HorizontalAlignment','right','VerticalAlignment','top', 'Rotation', angle)

set(gca,'XTickLabel','')  % 详解: 调用函数：set(gca,'XTickLabel','')

for i = 1:length(t)  % 详解: for 循环：迭代变量 i 遍历 1:length(t)
  ext(i,:) = get(t(i),'Extent');  % 详解: 调用函数：ext(i,:) = get(t(i),'Extent')
end  % 详解: 执行语句

LowYPoint = min(ext(:,2));  % 详解: 赋值：将 min(...) 的结果保存到 LowYPoint

if ~isempty(axis_label)  % 详解: 条件判断：if (~isempty(axis_label))
  Xl = get(gca, 'Xlim');  % 详解: 赋值：将 get(...) 的结果保存到 Xl
  XMidPoint = Xl(1)+abs(diff(Xl))/2;  % 详解: 赋值：将 Xl(...) 的结果保存到 XMidPoint
  tl = text(XMidPoint,LowYPoint, axis_label, 'VerticalAlignment','top', ...  % 详解: 赋值：将 text(...) 的结果保存到 tl
	    'HorizontalAlignment','center');  % 详解: 执行语句
end  % 详解: 执行语句




