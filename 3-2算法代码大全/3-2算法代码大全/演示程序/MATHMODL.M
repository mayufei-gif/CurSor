% 文件: MATHMODL.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clc;echo on;  % 详解: 执行语句
echo off  % 详解: 执行语句
more(25)  % 详解: 调用函数：more(25)
echo on  % 详解: 执行语句

pause  % 详解: 执行语句

type contents  % 详解: 执行语句
more off  % 详解: 执行语句

pause  % 详解: 执行语句
clc  % 详解: 执行语句


pause  % 详解: 执行语句
help plot  % 详解: 执行语句

pause  % 详解: 执行语句
pause  % 详解: 执行语句
help spline  % 详解: 执行语句

pause  % 详解: 执行语句
clear;close;  % 详解: 执行语句
x=[1 2 4 5];y=[1 3 4 2];  % 详解: 赋值：计算表达式并保存到 x
p=spline(x,y);  % 详解: 赋值：将 spline(...) 的结果保存到 p
xi=1:0.1:5;yi=ppval(p,xi);  % 详解: 赋值：计算表达式并保存到 xi
plot(x,y,'o',xi,yi,'k');  % 详解: 调用函数：plot(x,y,'o',xi,yi,'k')
title('not-a-knot SPLINE');  % 详解: 调用函数：title('not-a-knot SPLINE')
[b,c]=unmkpp(p)  % 详解: 执行语句


pause  % 详解: 执行语句
close;clc  % 详解: 执行语句

pause  % 详解: 执行语句
help constr  % 详解: 执行语句


pause  % 详解: 执行语句
x0=[10,10,10];  % 详解: 赋值：计算表达式并保存到 x0
options(13)=1;  % 详解: 执行语句
[x,options]=constr('optfun2',x0,options,[-inf,10,-inf]',[inf,20,inf]);  % 执行语句  % 详解: 执行语句  % 详解: 执行语句
x  % 详解: 执行语句
-options(8)  % 详解: 执行语句

pause  % 详解: 执行语句
clc  % 详解: 执行语句

pause  % 详解: 执行语句
help kruskal  % 详解: 执行语句

clear;map=[1 2 30;2 4 5;2 5 50;3 2 6;4 3 1;1 4 20;1 5 30]  % 详解: 执行语句
[out,len]=kruskal(map)  % 详解: 执行语句

pause  % 详解: 执行语句
close;clc;  % 详解: 执行语句

pause  % 详解: 执行语句
type jm98a1  % 详解: 执行语句
echo off  % 详解: 执行语句
jm98a1  % 详解: 执行语句
echo on  % 详解: 执行语句
pause  % 详解: 执行语句
type jm98a2  % 详解: 执行语句
echo off  % 详解: 执行语句
jm98a2  % 详解: 执行语句
echo on  % 详解: 执行语句
pause  % 详解: 执行语句
type jm98a1  % 详解: 执行语句
echo off  % 详解: 执行语句
jm98a3  % 详解: 执行语句
echo on  % 详解: 执行语句
pause  % 详解: 执行语句
clc  % 详解: 执行语句
echo off  % 详解: 执行语句




