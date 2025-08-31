% 文件: cn2shorf.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [P d]=cn2shorf(W,k1,k2,t1,t2)  % 详解: 函数定义：cn2shorf(W,k1,k2,t1,t2), 返回：P d
[p1 d1]=n2shorf(W,k1,t1);  % 详解: 执行语句
[p2 d2]=n2shorf(W,t1,t2);  % 详解: 执行语句
[p3 d3]=n2shorf(W,t2,k2);  % 详解: 执行语句
dt1=d1+d2+d3;  % 详解: 赋值：计算表达式并保存到 dt1
[p4 d4]=n2shorf(W,k1,t2);  % 详解: 执行语句
[p5 d5]=n2shorf(W,t2,t1);  % 详解: 执行语句
[p6 d6]=n2shorf(W,t1,k2);  % 详解: 执行语句
dt2=d4+d5+d6;  % 详解: 赋值：计算表达式并保存到 dt2
if dt1<dt2  % 详解: 条件判断：if (dt1<dt2)
    d=dt1;  % 详解: 赋值：计算表达式并保存到 d
    P=[p1 p2(2:length(p2)) p3(2:length(p3))];  % 详解: 赋值：计算表达式并保存到 P
else  % 详解: 条件判断：else 分支
    d=dt2;  % 详解: 赋值：计算表达式并保存到 d
    P=[p4 p5(2:length(p5)) p6(2:length(p6))];  % 详解: 赋值：计算表达式并保存到 P
end  % 详解: 执行语句
P;  % 详解: 执行语句
d;  % 详解: 执行语句



