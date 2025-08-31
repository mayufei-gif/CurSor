% 文件: xticklabel_rotate90.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function xticklabel_rotate90(XTick,varargin)  % 详解: 函数定义：xticklabel_rotate90(XTick,varargin)


if ~isnumeric(XTick)  % 详解: 条件判断：if (~isnumeric(XTick))
   error('XTICKLABEL_ROTATE90 requires a numeric input argument');  % 详解: 调用函数：error('XTICKLABEL_ROTATE90 requires a numeric input argument')
end  % 详解: 执行语句

XTick = XTick(:);  % 详解: 赋值：将 XTick(...) 的结果保存到 XTick

set(gca,'XTick',XTick,'XTickLabel','')  % 详解: 调用函数：set(gca,'XTick',XTick,'XTickLabel','')

xTickLabels = num2str(XTick);  % 详解: 赋值：将 num2str(...) 的结果保存到 xTickLabels

hxLabel = get(gca,'XLabel');  % 详解: 赋值：将 get(...) 的结果保存到 hxLabel
xLabelString = get(hxLabel,'String');  % 详解: 赋值：将 get(...) 的结果保存到 xLabelString

if ~isempty(xLabelString)  % 详解: 条件判断：if (~isempty(xLabelString))
   warning('You may need to manually reset the XLABEL vertical position')  % 详解: 调用函数：warning('You may need to manually reset the XLABEL vertical position')
end  % 详解: 执行语句

set(hxLabel,'Units','data');  % 详解: 调用函数：set(hxLabel,'Units','data')
xLabelPosition = get(hxLabel,'Position');  % 详解: 赋值：将 get(...) 的结果保存到 xLabelPosition
y = xLabelPosition(2);  % 详解: 赋值：将 xLabelPosition(...) 的结果保存到 y

y=repmat(y,size(XTick,1),1);  % 详解: 赋值：将 repmat(...) 的结果保存到 y
fs = get(gca,'fontsize');  % 详解: 赋值：将 get(...) 的结果保存到 fs

hText = text(XTick, y, xTickLabels,'fontsize',fs);  % 详解: 赋值：将 text(...) 的结果保存到 hText

set(hText,'Rotation',90,'HorizontalAlignment','right',varargin{:})  % 详解: 调用函数：set(hText,'Rotation',90,'HorizontalAlignment','right',varargin{:})





