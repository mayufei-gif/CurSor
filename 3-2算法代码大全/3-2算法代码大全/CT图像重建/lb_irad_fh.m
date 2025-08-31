% 文件: lb_irad_fh.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%% = = =主程序= = = %%
clc;  % 详解: 执行语句
clear all;  % 详解: 执行语句
close all;  % 详解: 执行语句
N = 256;  % 详解: 赋值：计算表达式并保存到 N
I = phantom(N);  % 详解: 赋值：将 phantom(...) 的结果保存到 I
delta = pi/180;  % 详解: 赋值：计算表达式并保存到 delta
theta = 0:1:179;  % 详解: 赋值：计算表达式并保存到 theta
theta_num = length(theta);  % 详解: 赋值：将 length(...) 的结果保存到 theta_num
d = 1;  % 详解: 赋值：计算表达式并保存到 d
P = radon(I,theta);  % 详解: 赋值：将 radon(...) 的结果保存到 P
[mm,nn] = size(P);  % 详解: 获取向量/矩阵尺寸
e = floor((mm-N-1)/2+1)+1;  % 详解: 赋值：将 floor(...) 的结果保存到 e
P = P(e:N+e-1,:);  % 详解: 赋值：将 P(...) 的结果保存到 P
P1 = reshape(P,N,theta_num);  % 详解: 赋值：将 reshape(...) 的结果保存到 P1

fh_RL = RLfilter(N,d);  % 详解: 赋值：将 RLfilter(...) 的结果保存到 fh_RL
fh_SL = SLfilter(N,d);  % 详解: 赋值：将 SLfilter(...) 的结果保存到 fh_SL
rec = Backprojection(theta_num,N,P1,delta);  % 详解: 赋值：将 Backprojection(...) 的结果保存到 rec

rec_RL = RLfilteredbackprojection(theta_num,N,P1,delta,fh_RL);  % 详解: 赋值：将 RLfilteredbackprojection(...) 的结果保存到 rec_RL

rec_SL = SLfilteredbackprojection(theta_num,N,P1,delta,fh_SL);  % 详解: 赋值：将 SLfilteredbackprojection(...) 的结果保存到 rec_SL

figure;  % 详解: 执行语句
subplot(2,2,1),imshow(I),xlabel('(a)256x256头模型（原始图像）');  % 详解: 调用函数：subplot(2,2,1),imshow(I),xlabel('(a)256x256头模型（原始图像）')
subplot(2,2,2),imshow(rec,[]),xlabel('(b)直接反投影重建图像');  % 详解: 调用函数：subplot(2,2,2),imshow(rec,[]),xlabel('(b)直接反投影重建图像')
subplot(2,2,3),imshow(rec_RL,[]),xlabel('(c)R-L函数滤波反投影重建图像');  % 详解: 调用函数：subplot(2,2,3),imshow(rec_RL,[]),xlabel('(c)R-L函数滤波反投影重建图像')
subplot(2,2,4),imshow(rec_SL,[]),xlabel('(d)S-L函数滤波反投影重建图像');  % 详解: 调用函数：subplot(2,2,4),imshow(rec_SL,[]),xlabel('(d)S-L函数滤波反投影重建图像')



