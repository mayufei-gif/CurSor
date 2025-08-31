% 文件: Primf.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [T c]=Primf(a)  % 详解: 函数定义：Primf(a), 返回：T c

l=length(a);  % 详解: 赋值：将 length(...) 的结果保存到 l
a(a==0)=inf;  % 详解: 执行语句
k=1:l;  % 详解: 赋值：计算表达式并保存到 k
listV(k)=0;  % 详解: 执行语句
listV(1)=1;  % 详解: 执行语句
e=1;  % 详解: 赋值：计算表达式并保存到 e
while (e<l)  % 详解: while 循环：当 ((e<l)) 为真时迭代
    min=inf;  % 详解: 赋值：计算表达式并保存到 min
    for i=1:l  % 详解: for 循环：迭代变量 i 遍历 1:l
        if listV(i)==1  % 详解: 条件判断：if (listV(i)==1)
            for j=1:l  % 详解: for 循环：迭代变量 j 遍历 1:l
                if listV(j)==0 & min>a(i,j)  % 详解: 条件判断：if (listV(j)==0 & min>a(i,j))
                        min=a(i,j);b=a(i,j);  % 详解: 赋值：将 a(...) 的结果保存到 min
                        s=i;d=j;  % 详解: 赋值：计算表达式并保存到 s
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    listV(d)=1;  % 详解: 执行语句
    distance(e)=b;  % 详解: 执行语句
    source(e)=s;  % 详解: 执行语句
    destination(e)=d;  % 详解: 执行语句
    e=e+1;  % 详解: 赋值：计算表达式并保存到 e
end  % 详解: 执行语句

T=[source;destination];  % 详解: 赋值：计算表达式并保存到 T
for g=1:e-1  % 详解: for 循环：迭代变量 g 遍历 1:e-1
    c(g)=a(T(1,g),T(2,g));  % 详解: 调用函数：c(g)=a(T(1,g),T(2,g))
end  % 详解: 执行语句
c;  % 详解: 执行语句



