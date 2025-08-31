% 文件: is_psd.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function b = positive_semidefinite(M)  % 详解: 执行语句

E = eig(M);  % 详解: 赋值：将 eig(...) 的结果保存到 E
if length(find(E>=0)) == length(E)  % 详解: 条件判断：if (length(find(E>=0)) == length(E))
  b = 1;  % 详解: 赋值：计算表达式并保存到 b
else  % 详解: 条件判断：else 分支
  b = 0;  % 详解: 赋值：计算表达式并保存到 b
end  % 详解: 执行语句




