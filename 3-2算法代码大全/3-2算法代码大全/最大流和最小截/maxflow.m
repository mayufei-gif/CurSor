% 文件: maxflow.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%function [f,s]=maxflow(startp,endp,c)
%c为容量网络
%对容量网络的填写做一下说明
%容量具有方向性，比如弧(i,j)的容量为10，弧(j,i)为0
%即矩阵无须有对称性
function [f,s]=maxflow(startp,endp,c)  % 详解: 函数定义：maxflow(startp,endp,c), 返回：f,s
n=length(c);  % 详解: 赋值：将 length(...) 的结果保存到 n
f=zeros(size(c));  % 详解: 赋值：将 zeros(...) 的结果保存到 f
l=zeros(1,n);d=zeros(1,n);examine=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 l
l(startp)=0.5;d(startp)=inf;  % 详解: 执行语句
while 1  % 详解: while 循环：当 (1) 为真时迭代
    ifexam=0;ifl=0;  % 详解: 赋值：计算表达式并保存到 ifexam
    for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
        if l(i)~=0  % 详解: 条件判断：if (l(i)~=0)
            ifl=ifl+1;  % 详解: 赋值：计算表达式并保存到 ifl
            if examine(i)==1  % 详解: 条件判断：if (examine(i)==1)
                ifexam=ifexam+1;  % 详解: 赋值：计算表达式并保存到 ifexam
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    if ifl==ifexam  % 详解: 条件判断：if (ifl==ifexam)
        break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
    for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
        if l(i)~=0&examine(i)==0  % 详解: 条件判断：if (l(i)~=0&examine(i)==0)
            break;  % 详解: 跳出循环：break
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    for j=1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
        if c(i,j)~=0  % 详解: 条件判断：if (c(i,j)~=0)
            if f(i,j)<c(i,j)&l(j)==0  % 详解: 条件判断：if (f(i,j)<c(i,j)&l(j)==0)
                l(j)=i;  % 详解: 执行语句
                d(j)=min(d(i),c(i,j)-f(i,j));  % 详解: 调用函数：d(j)=min(d(i),c(i,j)-f(i,j))
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        if c(j,i)~=0  % 详解: 条件判断：if (c(j,i)~=0)
            if f(j,i)>0&l(j)==0  % 详解: 条件判断：if (f(j,i)>0&l(j)==0)
                
                l(j)=-i;  % 详解: 执行语句
                d(j)=min(d(i),f(i,j));  % 详解: 调用函数：d(j)=min(d(i),f(i,j))
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    examine(i)=1;  % 详解: 执行语句
    if l(endp)~=0  % 详解: 条件判断：if (l(endp)~=0)
        j=endp;  % 详解: 赋值：计算表达式并保存到 j
        while 1  % 详解: while 循环：当 (1) 为真时迭代
            if l(j)~=0.5  % 详解: 条件判断：if (l(j)~=0.5)
                if l(j)>0  % 详解: 条件判断：if (l(j)>0)
                    i=l(j);  % 详解: 赋值：将 l(...) 的结果保存到 i
                    f(i,j)=f(i,j)+d(endp);  % 详解: 调用函数：f(i,j)=f(i,j)+d(endp)
                    j=i;  % 详解: 赋值：计算表达式并保存到 j
                end  % 详解: 执行语句
                if l(j)<0  % 详解: 条件判断：if (l(j)<0)
                    i=-l(j);  % 详解: 赋值：计算表达式并保存到 i
                    f(j,i)=f(j,i)-d(endp);  % 详解: 调用函数：f(j,i)=f(j,i)-d(endp)
                    j=i;  % 详解: 赋值：计算表达式并保存到 j
                end  % 详解: 执行语句
            else  % 详解: 条件判断：else 分支
                l=zeros(1,n);break;  % 详解: 赋值：将 zeros(...) 的结果保存到 l
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        l(startp)=0.5;d(startp)=inf;examine=zeros(1,n);  % 详解: 调用函数：l(startp)=0.5;d(startp)=inf;examine=zeros(1,n)
    end  % 详解: 执行语句
end  % 详解: 执行语句
s=[];ns=0;  % 详解: 赋值：计算表达式并保存到 s
for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
    if l(i)~=0  % 详解: 条件判断：if (l(i)~=0)
        ns=ns+1;  % 详解: 赋值：计算表达式并保存到 ns
        s(ns)=i;  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句
fprintf('f为最大可行流\n');  % 详解: 调用函数：fprintf('f为最大可行流\n')
fprintf('图的最小截划分得到的一个子集s为：\n');  % 详解: 调用函数：fprintf('图的最小截划分得到的一个子集s为：\n')
disp(s);  % 详解: 调用函数：disp(s)

    



