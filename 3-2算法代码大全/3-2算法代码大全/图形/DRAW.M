% 文件: DRAW.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function fig = draw()  % 详解: 执行语句


load draw  % 详解: 执行语句

h0 = figure('Color',[0.8 0.8 0.8], ...  % 详解: 赋值：将 figure(...) 的结果保存到 h0
	'Colormap',mat0, ...  % 详解: 执行语句
	'Name','Smooth Draw', ...  % 详解: 执行语句
	'NumberTitle','off', ...  % 详解: 执行语句
	'PointerShapeCData',mat1, ...  % 详解: 执行语句
	'Position',[40 45 560 375], ...  % 详解: 执行语句
	'Resize','off', ...  % 详解: 执行语句
	'Tag','drawfig', ...  % 详解: 执行语句
	'UserData',mat2);  % 详解: 执行语句
h1 = axes('Parent',h0, ...  % 详解: 赋值：将 axes(...) 的结果保存到 h1
	'Units','pixels', ...  % 详解: 执行语句
	'Box','on', ...  % 详解: 执行语句
	'CameraUpVector',[0 1 0], ...  % 详解: 执行语句
	'CameraUpVectorMode','manual', ...  % 详解: 执行语句
	'Color',[1 1 1], ...  % 详解: 执行语句
	'ColorOrder',mat3, ...  % 详解: 执行语句
	'HandleVisibility','callback', ...  % 详解: 执行语句
	'Position',[52 135 347 218], ...  % 详解: 执行语句
	'Tag','Axes1', ...  % 详解: 执行语句
	'XColor',[0 0 0], ...  % 详解: 执行语句
	'XGrid','on', ...  % 详解: 执行语句
	'XLimMode','manual', ...  % 详解: 执行语句
	'YColor',[0 0 0], ...  % 详解: 执行语句
	'YGrid','on', ...  % 详解: 执行语句
	'YLimMode','manual', ...  % 详解: 执行语句
	'ZColor',[0 0 0], ...  % 详解: 执行语句
	'ZGrid','on');  % 详解: 执行语句
h2 = text('Parent',h1, ...  % 详解: 赋值：将 text(...) 的结果保存到 h2
	'Color',[0 0 0], ...  % 详解: 执行语句
	'HandleVisibility','off', ...  % 详解: 执行语句
	'HorizontalAlignment','center', ...  % 详解: 执行语句
	'Position',[0.4971098265895954 -0.1105990783410138 9.160254037844386], ...  % 详解: 执行语句
	'Tag','Axes1Text4', ...  % 详解: 执行语句
	'VerticalAlignment','cap');  % 详解: 执行语句
set(get(h2,'Parent'),'XLabel',h2);  % 详解: 调用函数：set(get(h2,'Parent'),'XLabel',h2)
h2 = text('Parent',h1, ...  % 详解: 赋值：将 text(...) 的结果保存到 h2
	'Color',[0 0 0], ...  % 详解: 执行语句
	'HandleVisibility','off', ...  % 详解: 执行语句
	'HorizontalAlignment','center', ...  % 详解: 执行语句
	'Position',[-0.08959537572254335 0.4930875576036867 9.160254037844386], ...  % 详解: 执行语句
	'Rotation',90, ...  % 详解: 执行语句
	'Tag','Axes1Text3', ...  % 详解: 执行语句
	'VerticalAlignment','baseline');  % 详解: 执行语句
set(get(h2,'Parent'),'YLabel',h2);  % 详解: 调用函数：set(get(h2,'Parent'),'YLabel',h2)
h2 = text('Parent',h1, ...  % 详解: 赋值：将 text(...) 的结果保存到 h2
	'Color',[0 0 0], ...  % 详解: 执行语句
	'HandleVisibility','off', ...  % 详解: 执行语句
	'HorizontalAlignment','right', ...  % 详解: 执行语句
	'Position',[-0.1502890173410405 1.101382488479263 9.160254037844386], ...  % 详解: 执行语句
	'Tag','Axes1Text2', ...  % 详解: 执行语句
	'Visible','off');  % 详解: 执行语句
set(get(h2,'Parent'),'ZLabel',h2);  % 详解: 调用函数：set(get(h2,'Parent'),'ZLabel',h2)
h2 = text('Parent',h1, ...  % 详解: 赋值：将 text(...) 的结果保存到 h2
	'Color',[0 0 0], ...  % 详解: 执行语句
	'HandleVisibility','off', ...  % 详解: 执行语句
	'HorizontalAlignment','center', ...  % 详解: 执行语句
	'Position',[0.4971098265895954 1.032258064516129 9.160254037844386], ...  % 详解: 执行语句
	'Tag','Axes1Text1', ...  % 详解: 执行语句
	'VerticalAlignment','bottom');  % 详解: 执行语句
set(get(h2,'Parent'),'Title',h2);  % 详解: 调用函数：set(get(h2,'Parent'),'Title',h2)
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[325.5 3 85.5 263.25], ...  % 详解: 执行语句
	'Style','frame', ...  % 详解: 执行语句
	'Tag','Frame1');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'Callback','clear;[x,y]=ginput;', ...  % 详解: 执行语句
	'FontSize',14, ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[333 222 72 25.5], ...  % 详解: 执行语句
	'String','Ginput', ...  % 详解: 执行语句
	'Tag','Pushbutton1');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'Callback','pp=csaps(x,y,0.9999);fnplt(pp,''b'',2);', ...  % 详解: 执行语句
	'FontSize',14, ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[333.75 192 72 24.75], ...  % 详解: 执行语句
	'String','Draw It', ...  % 详解: 执行语句
	'Tag','Pushbutton2');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'Callback','gline', ...  % 详解: 执行语句
	'FontSize',14, ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[333.75 163.5 72 24.75], ...  % 详解: 执行语句
	'String','Line', ...  % 详解: 执行语句
	'Tag','Pushbutton3');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'Callback',mat4, ...  % 详解: 执行语句
	'FontSize',14, ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[334.5 102 72 25.5], ...  % 详解: 执行语句
	'String','Range', ...  % 详解: 执行语句
	'Tag','Pushbutton4');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'Callback','cla;axis([0 1 0 1]);hold off', ...  % 详解: 执行语句
	'FontSize',14, ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[334.5 13.5 71.25 26.25], ...  % 详解: 执行语句
	'String','Erase', ...  % 详解: 执行语句
	'Tag','Pushbutton5');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'Callback','grid', ...  % 详解: 执行语句
	'FontSize',14, ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[334.5 133.5 72 25.5], ...  % 详解: 执行语句
	'String','Grid', ...  % 详解: 执行语句
	'Tag','Pushbutton6');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[21.75 9 273 63.75], ...  % 详解: 执行语句
	'Style','frame', ...  % 详解: 执行语句
	'Tag','Frame2');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'BackgroundColor',[1 1 1], ...  % 详解: 执行语句
	'Callback','eval(get(gcbo,''string''))', ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[32.25 9.75 261 46.5], ...  % 详解: 执行语句
	'Style','edit', ...  % 详解: 执行语句
	'Tag','EditText1');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'Callback',mat5, ...  % 详解: 执行语句
	'FontSize',14, ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[334.5 73.5 71.25 23.25], ...  % 详解: 执行语句
	'String','Labels', ...  % 详解: 执行语句
	'Tag','Pushbutton7');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[27 57.75 81 12], ...  % 详解: 执行语句
	'String','Command Window', ...  % 详解: 执行语句
	'Style','text', ...  % 详解: 执行语句
	'Tag','StaticText1');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'BackgroundColor',[0.752941176470588 0.752941176470588 0.752941176470588], ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[332.25 248.25 75.75 15], ...  % 详解: 执行语句
	'String','Control Panel', ...  % 详解: 执行语句
	'Style','text', ...  % 详解: 执行语句
	'Tag','StaticText2');  % 详解: 执行语句
h1 = uicontrol('Parent',h0, ...  % 详解: 赋值：将 uicontrol(...) 的结果保存到 h1
	'Units','points', ...  % 详解: 执行语句
	'Callback','hold', ...  % 详解: 执行语句
	'FontSize',14, ...  % 详解: 执行语句
	'ListboxTop',0, ...  % 详解: 执行语句
	'Position',[334.5 43.5 70.5 24.75], ...  % 详解: 执行语句
	'String','Hold', ...  % 详解: 执行语句
	'Tag','Pushbutton8');  % 详解: 执行语句
if nargout > 0, fig = h0; end  % 详解: 条件判断：if (nargout > 0, fig = h0; end)




