% 文件: dengwen.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clear;close all;  % 详解: 执行语句
fphn=fopen('hunan.txt','r');  % 详解: 赋值：将 fopen(...) 的结果保存到 fphn
hnb=fgetl(fphn);  % 详解: 赋值：将 fgetl(...) 的结果保存到 hnb
hnmap=fscanf(fphn,'%f %f',[2,59]);  % 详解: 赋值：将 fscanf(...) 的结果保存到 hnmap
fclose(fphn);  % 详解: 调用函数：fclose(fphn)
hnmap=hnmap';  % 赋值：设置变量 hnmap  % 详解: 赋值：计算表达式并保存到 hnmap  % 详解: 赋值：计算表达式并保存到 hnmap
xa=hnmap(:,[1]);  % 详解: 赋值：将 hnmap(...) 的结果保存到 xa
ya=hnmap(:,[2]);  % 详解: 赋值：将 hnmap(...) 的结果保存到 ya

fp=fopen('LATLON57.txt','r');  % 详解: 赋值：将 fopen(...) 的结果保存到 fp
LL57=fscanf(fp,'%d %f %f',[3,97]);  % 详解: 赋值：将 fscanf(...) 的结果保存到 LL57
fclose(fp);  % 详解: 调用函数：fclose(fp)
LL57=LL57';  % 赋值：设置变量 LL57  % 详解: 赋值：计算表达式并保存到 LL57  % 详解: 赋值：计算表达式并保存到 LL57
x=LL57(:,[3])/10;  % 详解: 赋值：将 LL57(...) 的结果保存到 x
y=LL57(:,[2])/10;  % 详解: 赋值：将 LL57(...) 的结果保存到 y


fpy=fopen('etw00100.txt','r');  % 详解: 赋值：将 fopen(...) 的结果保存到 fpy
ymd57=fscanf(fpy,'%d',[3,1]);  % 详解: 赋值：将 fscanf(...) 的结果保存到 ymd57
yu97=fscanf(fpy,'%d %f %f',[3,97]);  % 详解: 赋值：将 fscanf(...) 的结果保存到 yu97
fclose(fpy);  % 详解: 调用函数：fclose(fpy)
yu97=yu97';  % 赋值：设置变量 yu97  % 详解: 赋值：计算表达式并保存到 yu97  % 详解: 赋值：计算表达式并保存到 yu97
z=yu97(:,[2]);  % 详解: 赋值：将 yu97(...) 的结果保存到 z

hold on;  % 详解: 执行语句
plot(xa,ya,'.','markersize',5,'color','red');  % 详解: 调用函数：plot(xa,ya,'.','markersize',5,'color','red')

plot(x,y,'.','markersize',6);  % 详解: 调用函数：plot(x,y,'.','markersize',6)
[xi,yi]=meshgrid(linspace(min(x),max(x),25),linspace(min(y),max(y),25));  % 详解: 统计：最大/最小值
zi=griddata(x,y,z,xi,yi,'cubic');  % 详解: 赋值：将 griddata(...) 的结果保存到 zi
hold on;[c,h]=contour(xi,yi,zi,'b-');  % 详解: 执行语句
clabel(c,h);hold off;  % 详解: 执行语句



