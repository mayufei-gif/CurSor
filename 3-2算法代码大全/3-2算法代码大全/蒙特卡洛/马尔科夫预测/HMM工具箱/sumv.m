% 文件: sumv.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function T2 = sumv(T1, sum_over)  % 详解: 执行语句

T2 = T1;  % 详解: 赋值：计算表达式并保存到 T2
for i=1:length(sum_over)  % 详解: for 循环：迭代变量 i 遍历 1:length(sum_over)
  if sum_over(i) <= ndims(T2)  % 详解: 条件判断：if (sum_over(i) <= ndims(T2))
    T2=sum(T2, sum_over(i));  % 详解: 赋值：将 sum(...) 的结果保存到 T2
  end  % 详解: 执行语句
end  % 详解: 执行语句
T2 = squeeze(T2);  % 详解: 赋值：将 squeeze(...) 的结果保存到 T2




