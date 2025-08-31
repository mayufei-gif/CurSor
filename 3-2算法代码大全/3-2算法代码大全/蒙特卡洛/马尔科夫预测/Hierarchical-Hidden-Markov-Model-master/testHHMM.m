% 文件: testHHMM.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clc;  % 详解: 执行语句
clear all;  % 详解: 执行语句
close all;  % 详解: 执行语句
addpath(genpath('../Graph'))  % 详解: 调用函数：addpath(genpath('../Graph'))

maxIter = 40;  % 详解: 赋值：计算表达式并保存到 maxIter
maxError = 1e-03;  % 详解: 赋值：计算表达式并保存到 maxError

q(:,:,1) = [1 0 0 0 0 0 0 0;  % 详解: 执行语句
            1 1 2 0 0 0 0 0;  % 详解: 执行语句
            1 1 2 1 1 2 0 0;  % 详解: 执行语句
            1 1 1 2 1 1 1 2];  % 详解: 执行语句
q(:,:,2) = [3 0 0 0 0 0 0 0;  % 详解: 执行语句
            3 3 0 0 0 0 0 0;  % 详解: 执行语句
            0 0 0 4 4 0 0 0;  % 详解: 执行语句
            0 0 0 0 0 0 0 0];  % 详解: 执行语句
        
testSeq = [ 1 3 1 3 2 5 8 4 2 5 4 6 7 2 8;  % 详解: 赋值：计算表达式并保存到 testSeq
            7 3 5 8 3 5 2 3 7 5 4 6 3 2 5];  % 详解: 执行语句
alphabet = 1:8;  % 详解: 赋值：计算表达式并保存到 alphabet

[prodY prodX] = find(q(:,:,1)==1 & q(:,:,2)==0);  % 详解: 执行语句
[allY allX] = find(q(:,:,1)==1);  % 详解: 执行语句
[intY intX] = find(q(:,:,2)~=0);  % 详解: 执行语句
PI = zeros(size(q,2),size(q,2),size(q,1)-1);  % 详解: 赋值：将 zeros(...) 的结果保存到 PI
for i=1:length(allX)  % 详解: for 循环：迭代变量 i 遍历 1:length(allX)
    if allY(i)~=1  % 详解: 条件判断：if (allY(i)~=1)
        parent_i = find(cumsum(q(allY(i)-1,:,2))>=allX(i),1);  % 详解: 赋值：将 find(...) 的结果保存到 parent_i
        PI(parent_i,allX(i),allY(i)-1)= 1+(rand(1)-0.5)/5;  % 详解: 生成随机数/矩阵
    end  % 详解: 执行语句
end  % 详解: 执行语句

A = zeros(size(q,2),size(q,2),size(q,1)-1);  % 详解: 赋值：将 zeros(...) 的结果保存到 A
for i=1:length(allX)  % 详解: for 循环：迭代变量 i 遍历 1:length(allX)
    if allY(i)~=1  % 详解: 条件判断：if (allY(i)~=1)
        parent_i = find(cumsum(q(allY(i)-1,:,2))>=allX(i),1);  % 详解: 赋值：将 find(...) 的结果保存到 parent_i
        jArray = find(PI(parent_i,:,allY(i)-1)~=0);  % 详解: 赋值：将 find(...) 的结果保存到 jArray
        jArray = [jArray jArray(end)+1];  % 详解: 赋值：计算表达式并保存到 jArray
        A(allX(i),jArray,allY(i)-1)= 1+(rand(size(jArray,2),1)-0.5)/5;  % 详解: 生成随机数/矩阵
    end  % 详解: 执行语句
end  % 详解: 执行语句

for i=1:length(prodX);  % 详解: for 循环：迭代变量 i 遍历 1:length(prodX);
    r = ones(1,length(alphabet))+(rand(1,length(alphabet))-0.5)/5;  % 详解: 赋值：将 ones(...) 的结果保存到 r
    B(prodY(i),prodX(i),1:length(alphabet)) = r/sum(r);  % 详解: 调用函数：B(prodY(i),prodX(i),1:length(alphabet)) = r/sum(r)
end;  % 详解: 执行语句

for zeile = 1:size(q,2)  % 详解: for 循环：迭代变量 zeile 遍历 1:size(q,2)
    for tiefe = 1:(size(q,1)-1)  % 详解: for 循环：迭代变量 tiefe 遍历 1:(size(q,1)-1)
        A(zeile,:,tiefe) = A(zeile,:,tiefe)/sum(A(zeile,:,tiefe));  % 详解: 调用函数：A(zeile,:,tiefe) = A(zeile,:,tiefe)/sum(A(zeile,:,tiefe))
        PI(zeile,:,tiefe) = PI(zeile,:,tiefe)/sum(PI(zeile,:,tiefe));  % 详解: 调用函数：PI(zeile,:,tiefe) = PI(zeile,:,tiefe)/sum(PI(zeile,:,tiefe))
    end  % 详解: 执行语句
end  % 详解: 执行语句
A(isnan(A))=0;  % 详解: 执行语句
PI(isnan(PI))=0;  % 详解: 执行语句

initA = A;  % 详解: 赋值：计算表达式并保存到 initA
initB = B;  % 详解: 赋值：计算表达式并保存到 initB
initPI = PI;  % 详解: 赋值：计算表达式并保存到 initPI

Palt = 0;  % 详解: 赋值：计算表达式并保存到 Palt
stop = 0;  % 详解: 赋值：计算表达式并保存到 stop
for iter = 1:maxIter  % 详解: for 循环：迭代变量 iter 遍历 1:maxIter

    ergA = zeros(size(initA));  % 详解: 赋值：将 zeros(...) 的结果保存到 ergA
    ergB = zeros(size(initB));  % 详解: 赋值：将 zeros(...) 的结果保存到 ergB
    ergPI = zeros(size(initPI));  % 详解: 赋值：将 zeros(...) 的结果保存到 ergPI

    ergAVis = zeros(size(initA));  % 详解: 赋值：将 zeros(...) 的结果保存到 ergAVis
    ergBVis = zeros(size(initB));  % 详解: 赋值：将 zeros(...) 的结果保存到 ergBVis
    ergPIVis = zeros(size(initPI));  % 详解: 赋值：将 zeros(...) 的结果保存到 ergPIVis

    Pact = 0;  % 详解: 赋值：计算表达式并保存到 Pact
    for s=1:2  % 详解: for 循环：迭代变量 s 遍历 1:2
        seq = testSeq(s,:);  % 详解: 赋值：将 testSeq(...) 的结果保存到 seq
        tic;  % 详解: 执行语句
            [PI, A, B, P] = HHMM_EM(q, seq, initA, initPI, initB, alphabet, 0);  % 详解: 执行语句
            Pact = Pact+P;  % 详解: 赋值：计算表达式并保存到 Pact
            B(isnan(B))=0;  % 详解: 执行语句
            A(isnan(A))=0;  % 详解: 执行语句
            PI(isnan(PI))=0;  % 详解: 执行语句
        toc;  % 详解: 执行语句

        ergA = ergA+A;  % 详解: 赋值：计算表达式并保存到 ergA
        ergB = ergB+B;  % 详解: 赋值：计算表达式并保存到 ergB
        ergPI = ergPI+PI;  % 详解: 赋值：计算表达式并保存到 ergPI
    end  % 详解: 执行语句

    for i=1:length(prodX);  % 详解: for 循环：迭代变量 i 遍历 1:length(prodX);
       ergBVis(prodY(i),prodX(i),:) = ergB(prodY(i),prodX(i),:)/sum(ergB(prodY(i),prodX(i),:));  % 详解: 调用函数：ergBVis(prodY(i),prodX(i),:) = ergB(prodY(i),prodX(i),:)/sum(ergB(prodY(i),prodX(i),:))
    end;  % 详解: 执行语句
    for zeile = 1:size(q,2)  % 详解: for 循环：迭代变量 zeile 遍历 1:size(q,2)
        for tiefe = 1:(size(q,1)-1)  % 详解: for 循环：迭代变量 tiefe 遍历 1:(size(q,1)-1)
            ergAVis(zeile,:,tiefe) = ergA(zeile,:,tiefe)/sum(ergA(zeile,:,tiefe));  % 详解: 调用函数：ergAVis(zeile,:,tiefe) = ergA(zeile,:,tiefe)/sum(ergA(zeile,:,tiefe))
            ergPIVis(zeile,:,tiefe) = ergPI(zeile,:,tiefe)/sum(ergPI(zeile,:,tiefe));  % 详解: 调用函数：ergPIVis(zeile,:,tiefe) = ergPI(zeile,:,tiefe)/sum(ergPI(zeile,:,tiefe))
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    ergAVis(isnan(ergAVis))=0;  % 详解: 执行语句
    ergPIVis(isnan(ergPIVis))=0;  % 详解: 执行语句
    ergBVis(isnan(ergBVis))=0;  % 详解: 执行语句
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 30 20])  % 详解: 调用函数：set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 30 20])
    drawHHMM(q, ergAVis, ergPIVis, ergBVis);  % 详解: 调用函数：drawHHMM(q, ergAVis, ergPIVis, ergBVis)
    axis tight;  % 详解: 执行语句
    V = AXIS;  % 详解: 赋值：计算表达式并保存到 V
    axis([V(1) V(2) V(3)-2 V(4)])  % 详解: 调用函数：axis([V(1) V(2) V(3)-2 V(4)])
    title(sprintf('Iteration %d',iter));  % 详解: 调用函数：title(sprintf('Iteration %d',iter))
    hold off;  % 详解: 执行语句
    drawnow;  % 详解: 执行语句
    
    if (abs(Pact-Palt)/(1+abs(Palt))) < maxError  % 详解: 条件判断：if ((abs(Pact-Palt)/(1+abs(Palt))) < maxError)
        states = q(:,:,1)==1;  % 详解: 赋值：将 q(...) 的结果保存到 states
        if norm(ergAVis(:)-initA(:),inf)/sum(states(:)) < maxError  % 详解: 条件判断：if (norm(ergAVis(:)-initA(:),inf)/sum(states(:)) < maxError)
            fprintf('Konvergenz nach %d Iterationen\n', iter)  % 详解: 调用函数：fprintf('Konvergenz nach %d Iterationen\n', iter)
            stop = 1;  % 详解: 赋值：计算表达式并保存到 stop
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    if stop == 0  % 详解: 条件判断：if (stop == 0)
        Palt = Pact;  % 详解: 赋值：计算表达式并保存到 Palt
        initA = ergAVis;  % 详解: 赋值：计算表达式并保存到 initA
        initB = ergBVis;  % 详解: 赋值：计算表达式并保存到 initB
        initPI = ergPIVis;  % 详解: 赋值：计算表达式并保存到 initPI
    else  % 详解: 条件判断：else 分支
        break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
end  % 详解: 执行语句



