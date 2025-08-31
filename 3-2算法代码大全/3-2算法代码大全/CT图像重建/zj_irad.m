% 文件: zj_irad.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clc;  % 详解: 执行语句
clear all;  % 详解: 执行语句
close all;  % 详解: 执行语句
N = 256;  % 详解: 赋值：计算表达式并保存到 N
I = phantom(N);  % 详解: 赋值：将 phantom(...) 的结果保存到 I
theta = 0:1:179;  % 详解: 赋值：计算表达式并保存到 theta
P = radon(I,theta);  % 详解: 赋值：将 radon(...) 的结果保存到 P
rc = iradon(P,theta,'linear','None');  % 详解: 赋值：将 iradon(...) 的结果保存到 rc
figure;  % 详解: 执行语句
imshow(I),title('原始图像');  % 详解: 调用函数：imshow(I),title('原始图像')
figure;  % 详解: 执行语句
imshow(rc,[]),title('直接反投影重建图像');  % 详解: 调用函数：imshow(rc,[]),title('直接反投影重建图像')




