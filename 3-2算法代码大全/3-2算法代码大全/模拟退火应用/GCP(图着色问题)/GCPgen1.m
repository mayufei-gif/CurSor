% 文件: GCPgen1.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [u,i,j,df]=GCPgen(e,d,b,n,w,lamda)  % 详解: 函数定义：GCPgen(e,d,b,n,w,lamda), 返回：u,i,j,df
u=1+fix(unifrnd(0,n));i=e(u);  % 详解: 赋值：计算表达式并保存到 u
while 1  % 详解: while 循环：当 (1) 为真时迭代
    j=1+fix(unifrnd(0,d+1));  % 详解: 赋值：计算表达式并保存到 j
    if j~=i  % 详解: 条件判断：if (j~=i)
        break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
end  % 详解: 执行语句

df=0;  % 详解: 赋值：计算表达式并保存到 df
for v=1:n  % 详解: for 循环：迭代变量 v 遍历 1:n
    if e(v)==j  % 详解: 条件判断：if (e(v)==j)
        df=df+b(v,u)*w(j);  % 详解: 赋值：计算表达式并保存到 df
    end  % 详解: 执行语句
    if e(v)==i  % 详解: 条件判断：if (e(v)==i)
        df=df-b(v,u)*w(i);  % 详解: 赋值：计算表达式并保存到 df
    end  % 详解: 执行语句
end  % 详解: 执行语句
df=w(j)-w(i)-lamda*df;  % 详解: 赋值：将 w(...) 的结果保存到 df




