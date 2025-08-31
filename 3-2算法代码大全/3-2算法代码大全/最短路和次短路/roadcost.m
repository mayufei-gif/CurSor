% 文件: roadcost.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function y=roadcost(road,c)  % 详解: 执行语句
y=0;  % 详解: 赋值：计算表达式并保存到 y
n=length(road);  % 详解: 赋值：将 length(...) 的结果保存到 n
for i=1:(n-1)  % 详解: for 循环：迭代变量 i 遍历 1:(n-1)
    y=y+c(road(i),road(i+1));  % 详解: 赋值：计算表达式并保存到 y
end  % 详解: 执行语句





