% 文件: floyd.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%%floyd代码%%%%%%
function [k,d,r,minC,minK]=floyd(w)  % 详解: 函数定义：floyd(w), 返回：k,d,r,minC,minK
n=size(w,1);d=w;  % 详解: 赋值：将 size(...) 的结果保存到 n
for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
    for j=1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
        r(i,j)=j;  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句
for k=1:n  % 详解: for 循环：迭代变量 k 遍历 1:n
    for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
        for j=1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
            if d(i,k)+d(k,j)<d(i,j)  % 详解: 条件判断：if (d(i,k)+d(k,j)<d(i,j))
                d(i,j)=d(i,k)+d(k,j);  % 详解: 调用函数：d(i,j)=d(i,k)+d(k,j)
                r(i,j)=r(i,k);  % 详解: 调用函数：r(i,j)=r(i,k)
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    sprintf ('%s','迭代次数k:'),k,sprintf('%s','迭代后距离矩阵d:'),d  % 详解: 执行语句
end  % 详解: 执行语句
sprintf ('%s','最优路径矩阵r:'),r,C=zeros(1,n);  % 详解: 调用函数：sprintf('%s','最优路径矩阵r:'),r,C=zeros(1,n)
for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
    for j=1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
        C(i)=C(i)+d(i,j);  % 详解: 调用函数：C(i)=C(i)+d(i,j)
    end  % 详解: 执行语句
end  % 详解: 执行语句
minC=C(1);minK=1;k=2;  % 详解: 赋值：将 C(...) 的结果保存到 minC
while k<n+1  % 详解: while 循环：当 (k<n+1) 为真时迭代
    if minC >C(k)  % 详解: 条件判断：if (minC >C(k))
        minC=C(k);minK=k;  % 详解: 赋值：将 C(...) 的结果保存到 minC
    elseif minC==C(k)  % 详解: 条件判断：elseif (minC==C(k))
            minK=[[minK],k];  % 详解: 赋值：计算表达式并保存到 minK
    end  % 详解: 执行语句
    k=k+1;  % 详解: 赋值：计算表达式并保存到 k
end  % 详解: 执行语句
    sprintf ('%s','最小总费用值minC:'),minC  % 详解: 执行语句
    sprintf ('%s','最优顶点编号minK:'),minK  % 详解: 执行语句





