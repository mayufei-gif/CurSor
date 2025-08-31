% 文件: max_mult.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function y=max_mult(A,x)  % 详解: 执行语句


if size(x,2)==1  % 详解: 条件判断：if (size(x,2)==1)
  X=x*ones(1,size(A,1));  % 详解: 赋值：计算表达式并保存到 X
  y=max(A'.*X)';  % 详解: 赋值：将 max(...) 的结果保存到 y
else  % 详解: 条件判断：else 分支
  X=repmat(x, [1 1 size(A,1)]);  % 详解: 赋值：将 repmat(...) 的结果保存到 X
  B=repmat(A, [1 1 size(x,2)]);  % 详解: 赋值：将 repmat(...) 的结果保存到 B
  C=permute(B,[2 3 1]);  % 详解: 赋值：将 permute(...) 的结果保存到 C
  y=permute(max(C.*X),[3 2 1]);  % 详解: 赋值：将 permute(...) 的结果保存到 y
end  % 详解: 执行语句




