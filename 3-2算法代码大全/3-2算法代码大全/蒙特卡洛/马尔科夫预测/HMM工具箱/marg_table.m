% 文件: marg_table.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function smallT = marg_table(bigT, bigdom, bigsz, onto, maximize)  % 详解: 执行语句

if nargin < 5, maximize = 0; end  % 详解: 条件判断：if (nargin < 5, maximize = 0; end)


smallT = myreshape(bigT, bigsz);  % 详解: 赋值：将 myreshape(...) 的结果保存到 smallT
sum_over = mysetdiff(bigdom, onto);  % 详解: 赋值：将 mysetdiff(...) 的结果保存到 sum_over
ndx = find_equiv_posns(sum_over, bigdom);  % 详解: 赋值：将 find_equiv_posns(...) 的结果保存到 ndx
if maximize  % 详解: 条件判断：if (maximize)
  for i=1:length(ndx)  % 详解: for 循环：迭代变量 i 遍历 1:length(ndx)
    smallT = max(smallT, [], ndx(i));  % 详解: 赋值：将 max(...) 的结果保存到 smallT
  end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  for i=1:length(ndx)  % 详解: for 循环：迭代变量 i 遍历 1:length(ndx)
    smallT = sum(smallT, ndx(i));  % 详解: 赋值：将 sum(...) 的结果保存到 smallT
  end  % 详解: 执行语句
end  % 详解: 执行语句


ns = zeros(1, max(bigdom));  % 详解: 赋值：将 zeros(...) 的结果保存到 ns
ns(bigdom) = bigsz;  % 详解: 执行语句

smallT = squeeze(smallT);  % 详解: 赋值：将 squeeze(...) 的结果保存到 smallT
smallT = myreshape(smallT, ns(onto));  % 详解: 赋值：将 myreshape(...) 的结果保存到 smallT




