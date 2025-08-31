% 文件: extend_domain_table.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function B = extend_domain_table(A, smalldom, smallsz, bigdom, bigsz)  % 详解: 执行语句

if isequal(size(A), [1 1])  % 详解: 条件判断：if (isequal(size(A), [1 1]))
  B = A;  % 详解: 赋值：计算表达式并保存到 B
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

map = find_equiv_posns(smalldom, bigdom);  % 详解: 赋值：将 find_equiv_posns(...) 的结果保存到 map
sz = ones(1, length(bigdom));  % 详解: 赋值：将 ones(...) 的结果保存到 sz
sz(map) = smallsz;  % 详解: 执行语句
B = myreshape(A, sz);  % 详解: 赋值：将 myreshape(...) 的结果保存到 B
sz = bigsz;  % 详解: 赋值：计算表达式并保存到 sz
sz(map) = 1;  % 详解: 执行语句
B = myrepmat(B, sz(:)');  % 赋值：设置变量 B  % 详解: 赋值：将 myrepmat(...) 的结果保存到 B  % 详解: 赋值：将 myrepmat(...) 的结果保存到 B
                           




