% 文件: myintersect.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function C = myintersect(A,B)  % 详解: 执行语句

A = A(:)'; B = B(:)';  % 详解: 赋值：将 A(...) 的结果保存到 A

if isempty(A)  % 详解: 条件判断：if (isempty(A))
  ma = 0;  % 详解: 赋值：计算表达式并保存到 ma
else  % 详解: 条件判断：else 分支
  ma = max(A);  % 详解: 赋值：将 max(...) 的结果保存到 ma
end  % 详解: 执行语句

if isempty(B)  % 详解: 条件判断：if (isempty(B))
  mb = 0;  % 详解: 赋值：计算表达式并保存到 mb
else  % 详解: 条件判断：else 分支
  mb = max(B);  % 详解: 赋值：将 max(...) 的结果保存到 mb
end  % 详解: 执行语句

if ma==0 | mb==0  % 详解: 条件判断：if (ma==0 | mb==0)
  C = [];  % 详解: 赋值：计算表达式并保存到 C
else  % 详解: 条件判断：else 分支
  bits = zeros(1, max(ma,mb));  % 详解: 赋值：将 zeros(...) 的结果保存到 bits
  bits(A) = 1;  % 详解: 执行语句
  C = B(logical(bits(B)));  % 详解: 赋值：将 B(...) 的结果保存到 C
end  % 详解: 执行语句





