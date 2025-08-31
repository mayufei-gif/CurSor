% 文件: logsumexpv.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function s = logsumexpv(a)  % 详解: 执行语句


a = a(:)'; % make row vector  % 详解: 赋值：将 a(...) 的结果保存到 a  % 详解: 赋值：将 a(...) 的结果保存到 a
m = max(a);  % 详解: 赋值：将 max(...) 的结果保存到 m
b = a - m*ones(1,length(a));  % 详解: 赋值：计算表达式并保存到 b
s = m + log(sum(exp(b)));  % 详解: 赋值：计算表达式并保存到 s





