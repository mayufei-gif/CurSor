% 文件: suptitle.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function hout=suptitle(str, fs)  % 详解: 执行语句




plotregion = .92;  % 详解: 赋值：计算表达式并保存到 plotregion

titleypos  = .95;  % 详解: 赋值：计算表达式并保存到 titleypos

if nargin < 2  % 详解: 条件判断：if (nargin < 2)
  fs = get(gcf,'defaultaxesfontsize')+4;  % 详解: 赋值：将 get(...) 的结果保存到 fs
end  % 详解: 执行语句

fudge=1;  % 详解: 赋值：计算表达式并保存到 fudge

haold = gca;  % 详解: 赋值：计算表达式并保存到 haold
figunits = get(gcf,'units');  % 详解: 赋值：将 get(...) 的结果保存到 figunits


	if (~strcmp(figunits,'pixels')),  % 详解: 条件判断：if ((~strcmp(figunits,'pixels')),)
		set(gcf,'units','pixels');  % 详解: 调用函数：set(gcf,'units','pixels')
		pos = get(gcf,'position');  % 详解: 赋值：将 get(...) 的结果保存到 pos
		set(gcf,'units',figunits);  % 详解: 调用函数：set(gcf,'units',figunits)
	else,  % 详解: 条件判断：else 分支
		pos = get(gcf,'position');  % 详解: 赋值：将 get(...) 的结果保存到 pos
	end  % 详解: 执行语句
	ff = (fs-4)*1.27*5/pos(4)*fudge;  % 详解: 赋值：计算表达式并保存到 ff




	
h = findobj(gcf,'Type','axes');  % 详解: 赋值：将 findobj(...) 的结果保存到 h


	


max_y=0;  % 详解: 赋值：计算表达式并保存到 max_y
min_y=1;  % 详解: 赋值：计算表达式并保存到 min_y

oldtitle =0;  % 详解: 赋值：计算表达式并保存到 oldtitle
for i=1:length(h),  % 详解: for 循环：迭代变量 i 遍历 1:length(h),
	if (~strcmp(get(h(i),'Tag'),'suptitle')),  % 详解: 条件判断：if ((~strcmp(get(h(i),'Tag'),'suptitle')),)
		pos=get(h(i),'pos');  % 详解: 赋值：将 get(...) 的结果保存到 pos
		if (pos(2) < min_y), min_y=pos(2)-ff/5*3;end;  % 详解: 条件判断：if ((pos(2) < min_y), min_y=pos(2)-ff/5*3;end;)
		if (pos(4)+pos(2) > max_y), max_y=pos(4)+pos(2)+ff/5*2;end;  % 详解: 条件判断：if ((pos(4)+pos(2) > max_y), max_y=pos(4)+pos(2)+ff/5*2;end;)
	else,  % 详解: 条件判断：else 分支
		oldtitle = h(i);  % 详解: 赋值：将 h(...) 的结果保存到 oldtitle
	end  % 详解: 执行语句
end  % 详解: 执行语句

if max_y > plotregion,  % 详解: 条件判断：if (max_y > plotregion,)
	scale = (plotregion-min_y)/(max_y-min_y);  % 详解: 赋值：计算表达式并保存到 scale
	for i=1:length(h),  % 详解: for 循环：迭代变量 i 遍历 1:length(h),
		pos = get(h(i),'position');  % 详解: 赋值：将 get(...) 的结果保存到 pos
		pos(2) = (pos(2)-min_y)*scale+min_y;  % 详解: 执行语句
		pos(4) = pos(4)*scale-(1-scale)*ff/5*3;  % 详解: 执行语句
		set(h(i),'position',pos);  % 详解: 调用函数：set(h(i),'position',pos)
	end  % 详解: 执行语句
end  % 详解: 执行语句

np = get(gcf,'nextplot');  % 详解: 赋值：将 get(...) 的结果保存到 np
set(gcf,'nextplot','add');  % 详解: 调用函数：set(gcf,'nextplot','add')
if (oldtitle),  % 详解: 条件判断：if ((oldtitle),)
	delete(oldtitle);  % 详解: 调用函数：delete(oldtitle)
end  % 详解: 执行语句
ha=axes('pos',[0 1 1 1],'visible','off','Tag','suptitle');  % 详解: 赋值：将 axes(...) 的结果保存到 ha
ht=text(.5,titleypos-1,str);set(ht,'horizontalalignment','center','fontsize',fs);  % 详解: 赋值：将 text(...) 的结果保存到 ht
set(gcf,'nextplot',np);  % 详解: 调用函数：set(gcf,'nextplot',np)
axes(haold);  % 详解: 调用函数：axes(haold)
if nargout,  % 详解: 条件判断：if (nargout,)
	hout=ht;  % 详解: 赋值：计算表达式并保存到 hout
end  % 详解: 执行语句







