% 文件: find_equiv_posns.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = find_equiv_posns(vsmall, vlarge)  % 详解: 执行语句
 

if isempty(vsmall) | isempty(vlarge)  % 详解: 条件判断：if (isempty(vsmall) | isempty(vlarge))
  p = [];  % 详解: 赋值：计算表达式并保存到 p
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句
  
bitvec = sparse(1, max(vlarge));  % 详解: 赋值：将 sparse(...) 的结果保存到 bitvec
bitvec(vsmall) = 1;  % 详解: 执行语句
p = find(bitvec(vlarge));  % 详解: 赋值：将 find(...) 的结果保存到 p





