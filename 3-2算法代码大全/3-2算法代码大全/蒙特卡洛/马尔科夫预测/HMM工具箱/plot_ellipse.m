% 文件: plot_ellipse.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% PLOT_ELLIPSE
% h=plot_ellipse(x,y,theta,a,b)
%
% This routine plots an ellipse with centre (x,y), axis lengths a,b
% with major axis at an angle of theta radians from the horizontal.  % 中文: 从水平的theta弧度角度的主要轴。 |||作者：P。Fieguth

%
% Author: P. Fieguth  % 中文: 98年1月||| http://ocho.uwaterloo.ca/~pfieguth/teaching/372/plot_ellipse.m
%         Jan. 98  % 中文: plot_matrix将2D矩阵作为灰度图像，并标记轴||| plot_matrix（m）|||对于0/1矩阵（例如邻接矩阵），请使用||| plot_matrix（M，1）|||图像（g）||| colormap（[1 1 1; 0 0 0]）;白色背景上的黑色正方形||| colormap（灰色）|||移动网格线，使它们不会与正方形相交|||关闭分数|||的令人困惑的标签理想情况下，我们可以将标签转移到轴线之间...
%
%http://ocho.uwaterloo.ca/~pfieguth/Teaching/372/plot_ellipse.m  % 中文: set（gca，'xticklabel'，[]）;

function h=plot_ellipse(x,y,theta,a,b)  % 详解: 执行语句

np = 100;  % 详解: 赋值：计算表达式并保存到 np
ang = [0:np]*2*pi/np;  % 详解: 赋值：计算表达式并保存到 ang
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];  % 详解: 赋值：计算表达式并保存到 R
pts = [x;y]*ones(size(ang)) + R*[cos(ang)*a; sin(ang)*b];  % 详解: 赋值：计算表达式并保存到 pts
h=plot( pts(1,:), pts(2,:) );  % 详解: 赋值：将 plot(...) 的结果保存到 h




