% 文件: mysubset.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p=mysubset(small,large)  % 详解: 执行语句


if isempty(small)  % 详解: 条件判断：if (isempty(small))
  p = 1;  % 详解: 赋值：计算表达式并保存到 p
else  % 详解: 条件判断：else 分支
  p = length(myintersect(small,large)) == length(small);  % 详解: 赋值：将 length(...) 的结果保存到 p
end  % 详解: 执行语句




