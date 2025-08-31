% 文件: DFS.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clear all;close all;clc  % 详解: 执行语句
b=[1 2;1 3;1 4;2 4;  % 详解: 赋值：计算表达式并保存到 b
3 6;4 6;4 7];  % 详解: 执行语句
c=b(:)  % 详解: 赋值：将 b(...) 的结果保存到 c
m=max(b(:));  % 详解: 赋值：将 max(...) 的结果保存到 m
A=compresstable2matrix(b)  % 详解: 赋值：将 compresstable2matrix(...) 的结果保存到 A
netplot(A,1)  % 详解: 调用函数：netplot(A,1)
title('原始网络拓扑图');  % 详解: 调用函数：title('原始网络拓扑图')
top=1;  % 详解: 赋值：计算表达式并保存到 top
stack(top)=1;  % 详解: 执行语句

flag=1;  % 详解: 赋值：计算表达式并保存到 flag
re=[];  % 详解: 赋值：计算表达式并保存到 re
while top~=0  % 详解: while 循环：当 (top~=0) 为真时迭代
    pre_len=length(stack);  % 详解: 赋值：将 length(...) 的结果保存到 pre_len
    i=stack(top);  % 详解: 赋值：将 stack(...) 的结果保存到 i
    for j=1:m  % 详解: for 循环：迭代变量 j 遍历 1:m
        if A(i,j)==1 && isempty(find(flag==j,1))  % 详解: 条件判断：if (A(i,j)==1 && isempty(find(flag==j,1)))
            top=top+1;  % 详解: 赋值：计算表达式并保存到 top
            stack(top)=j;  % 详解: 执行语句
            flag=[flag j];  % 详解: 赋值：计算表达式并保存到 flag
            re=[re;i j];  % 详解: 赋值：计算表达式并保存到 re
            break;  % 详解: 跳出循环：break
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    if length(stack)==pre_len  % 详解: 条件判断：if (length(stack)==pre_len)
        stack(top)=[];  % 详解: 执行语句
        top=top-1;  % 详解: 赋值：计算表达式并保存到 top
    end  % 详解: 执行语句
end  % 详解: 执行语句

A=compresstable2matrix(re);  % 详解: 赋值：将 compresstable2matrix(...) 的结果保存到 A
figure;  % 详解: 执行语句
netplot(A,1)  % 详解: 调用函数：netplot(A,1)
title('深度优先网络拓扑图');  % 详解: 调用函数：title('深度优先网络拓扑图')



