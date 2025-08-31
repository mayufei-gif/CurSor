% 文件: rotate_xlabel.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function hText = rotate_xlabel(degrees, newlabels)  % 详解: 执行语句


xtl = get(gca,'XTickLabel');  % 详解: 赋值：将 get(...) 的结果保存到 xtl
set(gca,'XTickLabel','');  % 详解: 调用函数：set(gca,'XTickLabel','')
lxtl = length(xtl);  % 详解: 赋值：将 length(...) 的结果保存到 lxtl
xtl = newlabels;  % 详解: 赋值：计算表达式并保存到 xtl
if 0  % 详解: 条件判断：if (0)
    lnl = length(newlabels);  % 详解: 赋值：将 length(...) 的结果保存到 lnl
    if lnl~=lxtl  % 详解: 条件判断：if (lnl~=lxtl)
        error('Number of new labels must equal number of old');  % 详解: 调用函数：error('Number of new labels must equal number of old')
    end;  % 详解: 执行语句
    xtl = newlabels;  % 详解: 赋值：计算表达式并保存到 xtl
end;  % 详解: 执行语句


hxLabel=get(gca,'XLabel');  % 详解: 赋值：将 get(...) 的结果保存到 hxLabel
xLP=get(hxLabel,'Position');  % 详解: 赋值：将 get(...) 的结果保存到 xLP
y=xLP(2);  % 详解: 赋值：将 xLP(...) 的结果保存到 y
XTick=get(gca,'XTick');  % 详解: 赋值：将 get(...) 的结果保存到 XTick
y=repmat(y,length(XTick),1);  % 详解: 赋值：将 repmat(...) 的结果保存到 y
fs = 12;  % 详解: 赋值：计算表达式并保存到 fs
hText=text(XTick,y,xtl,'fontsize',fs);  % 详解: 赋值：将 text(...) 的结果保存到 hText
set(hText,'Rotation',degrees,'HorizontalAlignment','right');  % 详解: 调用函数：set(hText,'Rotation',degrees,'HorizontalAlignment','right')


ylim = get(gca,'ylim');  % 详解: 赋值：将 get(...) 的结果保存到 ylim
height = ylim(2)-ylim(1);  % 详解: 赋值：将 ylim(...) 的结果保存到 height
N = length(hText);  % 详解: 赋值：将 length(...) 的结果保存到 N
for i=1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
  voffset = ylim(2) - 0*height;  % 详解: 赋值：将 ylim(...) 的结果保存到 voffset
  set(hText(i), 'position', [i voffset 0]);  % 详解: 调用函数：set(hText(i), 'position', [i voffset 0])
end  % 详解: 执行语句




