% 文件: BFS.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clear all;close all;clc  % 详解: 执行语句
b=[1 2;1 3;1 4;2 4;3 6;4 6;4 7];  % 详解: 赋值：计算表达式并保存到 b

m=max(b(:));  % 详解: 赋值：将 max(...) 的结果保存到 m
A=compresstable2matrix(b);  % 详解: 赋值：将 compresstable2matrix(...) 的结果保存到 A
netplot(A,1)  % 详解: 调用函数：netplot(A,1)

head=1;  % 详解: 赋值：计算表达式并保存到 head
tail=1;  % 详解: 赋值：计算表达式并保存到 tail
queue(head)=1;  % 详解: 执行语句
head=head+1;  % 详解: 赋值：计算表达式并保存到 head

flag=1;  % 详解: 赋值：计算表达式并保存到 flag
re=[];  % 详解: 赋值：计算表达式并保存到 re
while tail~=head  % 详解: while 循环：当 (tail~=head) 为真时迭代
    i=queue(tail);  % 详解: 赋值：将 queue(...) 的结果保存到 i
    for j=1:m  % 详解: for 循环：迭代变量 j 遍历 1:m
        if A(i,j)==1 && isempty(find(flag==j,1))  % 详解: 条件判断：if (A(i,j)==1 && isempty(find(flag==j,1)))
            queue(head)=j;  % 详解: 执行语句
            head=head+1;  % 详解: 赋值：计算表达式并保存到 head
            flag=[flag j];  % 详解: 赋值：计算表达式并保存到 flag
            re=[re;i j];  % 详解: 赋值：计算表达式并保存到 re
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    tail=tail+1;  % 详解: 赋值：计算表达式并保存到 tail
end  % 详解: 执行语句

A=compresstable2matrix(re);  % 详解: 赋值：将 compresstable2matrix(...) 的结果保存到 A
figure;  % 详解: 执行语句
netplot(A,1)  % 详解: 调用函数：netplot(A,1)



