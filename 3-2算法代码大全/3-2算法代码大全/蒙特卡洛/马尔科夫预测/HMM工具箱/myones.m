% 文件: myones.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function T = myones(sizes)  % 详解: 执行语句

if length(sizes)==0  % 详解: 条件判断：if (length(sizes)==0)
  T = 1;  % 详解: 赋值：计算表达式并保存到 T
elseif length(sizes)==1  % 详解: 条件判断：elseif (length(sizes)==1)
  T = ones(sizes, 1);  % 详解: 赋值：将 ones(...) 的结果保存到 T
else  % 详解: 条件判断：else 分支
  T = ones(sizes(:)');  % 创建全 1 矩阵/数组  % 详解: 赋值：将 ones(...) 的结果保存到 T  % 详解: 赋值：将 ones(...) 的结果保存到 T
end  % 详解: 执行语句




