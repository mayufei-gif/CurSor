% 文件: Malthus_test.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% Malthus模型  1790-1900年
function Malthus_test  % 详解: 执行语句
clc  % 详解: 执行语句
clear all  % 详解: 执行语句
xdata = [1790 1800 1810 1820 1830 1840 1850 1860 1870 1880 1890 1900];  % 详解: 赋值：计算表达式并保存到 xdata
ydata = [3.9 5.3 7.2 9.6 12.9 17.1 23.2 31.4 38.6 50.2 62.9 76.0];  % 详解: 赋值：计算表达式并保存到 ydata
r0 = 0.25;  % 详解: 赋值：计算表达式并保存到 r0
[r, resnorm] = lsqcurvefit(@malthus,r0,xdata,ydata)  % 详解: 执行语句
 
yfit = 3.9*exp(r*(xdata-1790));  % 详解: 赋值：计算表达式并保存到 yfit
plot(xdata,ydata,'*b',xdata,yfit,'-r')  % 详解: 调用函数：plot(xdata,ydata,'*b',xdata,yfit,'-r')
 






