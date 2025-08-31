% 文件: mult_by_table.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function bigT = mult_by_table(bigT, bigdom, bigsz, smallT, smalldom, smallsz)  % 详解: 执行语句

Ts = extend_domain_table(smallT, smalldom, smallsz, bigdom, bigsz);  % 详解: 赋值：将 extend_domain_table(...) 的结果保存到 Ts
bigT(:) = bigT(:) .* Ts(:);  % 详解: 调用函数：bigT(:) = bigT(:) .* Ts(:)




