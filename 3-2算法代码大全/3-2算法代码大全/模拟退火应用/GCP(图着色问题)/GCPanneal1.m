% 文件: GCPanneal1.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%function [e,c]=GCPanneal1(L,s,t,dt,lamda,d,b,n)
%
%图着色问题(Graph Colouring Problem)的退火算法
%GCP问题可看为将顶点集划分为最少个数独立集的问题
%
%求解此问题有两种算法，
%GCPanneal1适用于度数小于20的情形
%GCPanneal2适用于各种度数
%在GCPanneal1中，w(i)表示赋予颜色i的权值
%
%n为问题规模，即节点个数；b为关联矩阵
%lamda是一个大于1的罚函数因子
%d为图G的最大度数，最小着色上界为d+1
%e(u)表示u被着的颜色号
%c(i)表示着以颜色i的顶点个数
%
%L可取较大值，如500、1000；
%s取1、2等；t为初始温度，参考范围为0.5--2；
%dt为衰减因子，一般不小于0.9;
%L、s、t、dt应通过多次试验来确定，以获得优化的结果
%参考《非数值并行算法--模拟退火算法》科学出版社

function [e,c]=GCPanneal1(L,s,t,dt,lamda,d,b,n)  % 详解: 函数定义：GCPanneal1(L,s,t,dt,lamda,d,b,n), 返回：e,c
w(1)=2^d;  % 详解: 执行语句
for j=2:(d+1)  % 详解: for 循环：迭代变量 j 遍历 2:(d+1)
    w(j)=2*w(j-1)-w(1)-1;  % 详解: 执行语句
end  % 详解: 执行语句

e=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 e
e=e+1;  % 详解: 赋值：计算表达式并保存到 e
c=zeros(1,d+1);  % 详解: 赋值：将 zeros(...) 的结果保存到 c
c(1)=n;  % 详解: 执行语句

s0=0;  % 详解: 赋值：计算表达式并保存到 s0
while 1  % 详解: while 循环：当 (1) 为真时迭代
    a=0;  % 详解: 赋值：计算表达式并保存到 a
    for k=1:L  % 详解: for 循环：迭代变量 k 遍历 1:L
        [u,i,j,df]=GCPgen1(e,d,b,n,w,lamda);  % 详解: 执行语句
        if GCPacc1(df,t)  % 详解: 条件判断：if (GCPacc1(df,t))
              e(u)=j;c(i)=c(i)-1;c(j)=c(j)+1;  % 详解: 执行语句
              a=1;  % 详解: 赋值：计算表达式并保存到 a
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    fprintf('图中各点被着的颜色号\n');  % 详解: 调用函数：fprintf('图中各点被着的颜色号\n')
    disp(e);  % 详解: 调用函数：disp(e)
    fprintf('着以各颜色的顶点个数\n');  % 详解: 调用函数：fprintf('着以各颜色的顶点个数\n')
    disp(c);  % 详解: 调用函数：disp(c)
    t=t*dt  % 详解: 赋值：计算表达式并保存到 t
    if a==0  % 详解: 条件判断：if (a==0)
        s0=s0+1;  % 详解: 赋值：计算表达式并保存到 s0
    else  % 详解: 条件判断：else 分支
        s0=0;  % 详解: 赋值：计算表达式并保存到 s0
    end  % 详解: 执行语句
    if s0==s  % 详解: 条件判断：if (s0==s)
        break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
end  % 详解: 执行语句








