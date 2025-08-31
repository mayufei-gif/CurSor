% 文件: mk_unit_norm.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function B = mk_unit_norm(A)  % 详解: 执行语句


[nrows ncols] = size(A);  % 详解: 获取向量/矩阵尺寸
s = sum(A.^2);  % 详解: 赋值：将 sum(...) 的结果保存到 s
ndx = find(s==0);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
s(ndx)=1;  % 详解: 执行语句
B = A ./ repmat(sqrt(s), [nrows 1]);  % 详解: 赋值：计算表达式并保存到 B





