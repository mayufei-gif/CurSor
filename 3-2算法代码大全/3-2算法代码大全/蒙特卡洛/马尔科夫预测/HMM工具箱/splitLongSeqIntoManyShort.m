% 文件: splitLongSeqIntoManyShort.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function short = splitLongSeqIntoManyShort(long, Tsmall)  % 详解: 执行语句

T = length(long);  % 详解: 赋值：将 length(...) 的结果保存到 T
Nsmall = ceil(T/Tsmall);  % 详解: 赋值：将 ceil(...) 的结果保存到 Nsmall
short = cell(Nsmall,1);  % 详解: 赋值：将 cell(...) 的结果保存到 short

t = 1;  % 详解: 赋值：计算表达式并保存到 t
for i=1:Nsmall  % 详解: for 循环：迭代变量 i 遍历 1:Nsmall
  short{i} = long(:,t:min(T,t+Tsmall-1));  % 详解: 统计：最大/最小值
  t = t+Tsmall;  % 详解: 赋值：计算表达式并保存到 t
end  % 详解: 执行语句




