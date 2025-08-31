% 文件: normal.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%产生n个N(a,b)正态分布随机数
%其中a为均值，b为方差
%function x=normal(a,b,n)  % 中文: a = [11 4 0.2; 22 3 0.5; 0 3 0.4];
function x=normal(a,b,n)  % 详解: 执行语句
m=48;  % 详解: 赋值：计算表达式并保存到 m
for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
    r=rand(1,m);  % 详解: 赋值：将 rand(...) 的结果保存到 r
    x(i)=a+sqrt(b)*(sum(r)-m/2)/sqrt(m/12);  % 详解: 调用函数：x(i)=a+sqrt(b)*(sum(r)-m/2)/sqrt(m/12)
end  % 详解: 执行语句




