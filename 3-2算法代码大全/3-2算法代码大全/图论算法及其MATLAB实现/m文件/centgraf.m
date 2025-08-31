% 文件: centgraf.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [d0 d]=centgraf(W,A)  % 详解: 函数定义：centgraf(W,A), 返回：d0 d

n=length(W);  % 详解: 赋值：将 length(...) 的结果保存到 n
U=W;  % 详解: 赋值：计算表达式并保存到 U
m=1;  % 详解: 赋值：计算表达式并保存到 m
while m<=n  % 详解: while 循环：当 (m<=n) 为真时迭代
for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
    for j=1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
        if U(i,j)>U(i,m)+U(m,j)  % 详解: 条件判断：if (U(i,j)>U(i,m)+U(m,j))
            U(i,j)=U(i,m)+U(m,j);  % 详解: 调用函数：U(i,j)=U(i,m)+U(m,j)
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句
m=m+1;  % 详解: 赋值：计算表达式并保存到 m
end  % 详解: 执行语句

d1=max(U,[],2);  % 详解: 赋值：将 max(...) 的结果保存到 d1
d0t=min(d1);  % 详解: 赋值：将 min(...) 的结果保存到 d0t
d0=find(d1==min(d1));  % 详解: 赋值：将 find(...) 的结果保存到 d0

dt=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 dt
for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
    dt(i)=dot(U(i,:),A);  % 详解: 调用函数：dt(i)=dot(U(i,:),A)
end  % 详解: 执行语句
d=find(dt==min(dt));  % 详解: 赋值：将 find(...) 的结果保存到 d
ddt=min(dt);  % 详解: 赋值：将 min(...) 的结果保存到 ddt




