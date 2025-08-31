% 文件: previewfig.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function f = previewfig(h,varargin)  % 详解: 执行语句


filename = [tempname, '.png'];  % 详解: 赋值：计算表达式并保存到 filename
args = {'resolution',0,'format','png'};  % 详解: 赋值：计算表达式并保存到 args
if nargin > 1  % 详解: 条件判断：if (nargin > 1)
  exportfig(h, filename, varargin{:}, args{:});  % 详解: 调用函数：exportfig(h, filename, varargin{:}, args{:})
else  % 详解: 条件判断：else 分支
  exportfig(h, filename, args{:});  % 详解: 调用函数：exportfig(h, filename, args{:})
end  % 详解: 执行语句

X = imread(filename,'png');  % 详解: 赋值：将 imread(...) 的结果保存到 X
height = size(X,1);  % 详解: 赋值：将 size(...) 的结果保存到 height
width = size(X,2);  % 详解: 赋值：将 size(...) 的结果保存到 width
delete(filename);  % 详解: 调用函数：delete(filename)
f = figure( 'Name', 'Preview', ...  % 详解: 赋值：将 figure(...) 的结果保存到 f
	    'Menubar', 'none', ...  % 详解: 执行语句
	    'NumberTitle', 'off', ...  % 详解: 执行语句
	    'Visible', 'off');  % 详解: 执行语句
image(X);  % 详解: 调用函数：image(X)
axis image;  % 详解: 执行语句
ax = findobj(f, 'type', 'axes');  % 详解: 赋值：将 findobj(...) 的结果保存到 ax
axesPos = [0 0 width height];  % 详解: 赋值：计算表达式并保存到 axesPos
set(ax, 'Units', 'pixels', ...  % 详解: 执行语句
	'Position', axesPos, ...  % 详解: 执行语句
	'Visible', 'off');  % 详解: 执行语句
figPos = get(f,'Position');  % 详解: 赋值：将 get(...) 的结果保存到 figPos
rootSize = get(0,'ScreenSize');  % 详解: 赋值：将 get(...) 的结果保存到 rootSize
figPos(3:4) = axesPos(3:4);  % 详解: 调用函数：figPos(3:4) = axesPos(3:4)
if figPos(1) + figPos(3) > rootSize(3)  % 详解: 条件判断：if (figPos(1) + figPos(3) > rootSize(3))
  figPos(1) = rootSize(3) - figPos(3) - 50;  % 详解: 执行语句
end  % 详解: 执行语句
if figPos(2) + figPos(4) > rootSize(4)  % 详解: 条件判断：if (figPos(2) + figPos(4) > rootSize(4))
  figPos(2) = rootSize(4) - figPos(4) - 50;  % 详解: 执行语句
end  % 详解: 执行语句
set(f, 'Position',figPos, ...  % 详解: 执行语句
       'Visible', 'on');  % 详解: 执行语句




