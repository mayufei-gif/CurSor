% 文件: myrand.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function T = myrand(sizes)  % 详解: 执行语句

if length(sizes)==0  % 详解: 条件判断：if (length(sizes)==0)
  warning('myrand[]');  % 详解: 调用函数：warning('myrand[]')
  T = rand(1,1);  % 详解: 赋值：将 rand(...) 的结果保存到 T
elseif length(sizes)==1  % 详解: 条件判断：elseif (length(sizes)==1)
  T = rand(sizes, 1);  % 详解: 赋值：将 rand(...) 的结果保存到 T
else  % 详解: 条件判断：else 分支
  T = rand(sizes);  % 详解: 赋值：将 rand(...) 的结果保存到 T
end  % 详解: 执行语句




