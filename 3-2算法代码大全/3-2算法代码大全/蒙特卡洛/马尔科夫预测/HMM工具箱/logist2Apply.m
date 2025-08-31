% 文件: logist2Apply.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = logist2Apply(beta, x)  % 详解: 执行语句

[D Ncases] = size(x);  % 详解: 获取向量/矩阵尺寸
if length(beta)==D+1  % 详解: 条件判断：if (length(beta)==D+1)
  F = [x; ones(1,Ncases)];  % 详解: 赋值：计算表达式并保存到 F
else  % 详解: 条件判断：else 分支
  F = x;  % 详解: 赋值：计算表达式并保存到 F
end  % 详解: 执行语句
p = 1./(1+exp(-beta(:)'*F));  % 赋值：设置变量 p  % 详解: 赋值：计算表达式并保存到 p  % 详解: 赋值：计算表达式并保存到 p




