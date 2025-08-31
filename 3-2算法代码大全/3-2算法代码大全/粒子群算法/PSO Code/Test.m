% 文件: Test.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clear;  % 详解: 执行语句
results=0;  % 详解: 赋值：计算表达式并保存到 results
M=50 ;  % 详解: 赋值：计算表达式并保存到 M
his=zeros((M+2),1);  % 详解: 赋值：将 zeros(...) 的结果保存到 his

for i=1:M  % 详解: for 循环：迭代变量 i 遍历 1:M
    his(i)=PSO();  % 详解: 调用函数：his(i)=PSO()
    results = results+ his(i);  % 详解: 赋值：计算表达式并保存到 results
end  % 详解: 执行语句
avg= results/M  % 详解: 赋值：计算表达式并保存到 avg
Std=std(his(1:M))  % 详解: 赋值：将 std(...) 的结果保存到 Std
his=[his',avg,Std]  % 赋值：设置变量 his  % 详解: 赋值：计算表达式并保存到 his  % 详解: 赋值：计算表达式并保存到 his




