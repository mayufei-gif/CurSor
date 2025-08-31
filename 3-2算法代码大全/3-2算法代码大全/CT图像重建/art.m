% 文件: art.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clc;  % 详解: 执行语句
clear all;  % 详解: 执行语句
close all;  % 详解: 执行语句
N = 180;  % 详解: 赋值：计算表达式并保存到 N
N2 = N^2;  % 详解: 赋值：计算表达式并保存到 N2
I = phantom(N);  % 详解: 赋值：将 phantom(...) 的结果保存到 I
theta = linspace(0,180,181);  % 详解: 赋值：将 linspace(...) 的结果保存到 theta
theta = theta(1:180);  % 详解: 赋值：将 theta(...) 的结果保存到 theta
P_num = 260;  % 详解: 赋值：计算表达式并保存到 P_num
P = ParallelBeam(theta ,N ,P_num);  % 详解: 赋值：将 ParallelBeam(...) 的结果保存到 P
delta = 1;  % 详解: 赋值：计算表达式并保存到 delta
[W_ind,W_dat] = SystemMatrix(theta,N,P_num,delta);  % 详解: 执行语句
F = zeros(N2,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 F
lambda = 0.25;  % 详解: 赋值：计算表达式并保存到 lambda
c = 0;  % 详解: 赋值：计算表达式并保存到 c
irt_num = 5;  % 详解: 赋值：计算表达式并保存到 irt_num
while(c<irt_num)  % 详解: 调用函数：while(c<irt_num)
    for j = 1:length(theta)  % 详解: for 循环：迭代变量 j 遍历 1:length(theta)
        for i = 1:1:P_num  % 详解: for 循环：迭代变量 i 遍历 1:1:P_num
            u = W_ind((j-1)*P_num + i,:);  % 详解: 赋值：将 W_ind(...) 的结果保存到 u
            v = W_dat((j-1)*P_num + i,:);  % 详解: 赋值：将 W_dat(...) 的结果保存到 v
            
            if any(u) == 0  % 详解: 条件判断：if (any(u) == 0)
                continue;  % 详解: 继续下一次循环：continue
            end  % 详解: 执行语句
            w = zeros(1,N2);  % 详解: 赋值：将 zeros(...) 的结果保存到 w
            ind = u > 0;  % 详解: 赋值：计算表达式并保存到 ind
            w(u(ind))=v(ind);  % 详解: 调用函数：w(u(ind))=v(ind)
            PP = w * F;  % 详解: 赋值：计算表达式并保存到 PP
            C = (P(i,j)-PP)/sum(w.^2) * w';  % 修正项             % 详解: 赋值：计算表达式并保存到 C  % 详解: 赋值：计算表达式并保存到 C
            F = F + lambda * C;  % 详解: 赋值：计算表达式并保存到 F
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    F(F<0) = 0;  % 详解: 执行语句
    c = c+1;  % 详解: 赋值：计算表达式并保存到 c
end  % 详解: 执行语句
F = reshape(F,N,N)'; % 转换成N x N的图像矩阵  % 详解: 赋值：将 reshape(...) 的结果保存到 F  % 详解: 赋值：将 reshape(...) 的结果保存到 F
figure(1);  % 详解: 调用函数：figure(1)
imshow(I);xlabel('(a)180x180头模型图像');  % 详解: 调用函数：imshow(I);xlabel('(a)180x180头模型图像')
figure(2);  % 详解: 调用函数：figure(2)
imshow(F,[]);xlabel('(b)ART算法重建的图像');  % 详解: 调用函数：imshow(F,[]);xlabel('(b)ART算法重建的图像')
     




