% 文件: rectintSparseC.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [overlap, normoverlap] = rectintSparseC(A,B)  % 详解: 函数定义：rectintSparseC(A,B), 返回：overlap, normoverlap

if isempty(A) | isempty(B)  % 详解: 条件判断：if (isempty(A) | isempty(B))
  overlap = [];  % 详解: 赋值：计算表达式并保存到 overlap
  normoverlap = [];  % 详解: 赋值：计算表达式并保存到 normoverlap
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

leftA = A(:,1);  % 详解: 赋值：将 A(...) 的结果保存到 leftA
bottomA = A(:,2);  % 详解: 赋值：将 A(...) 的结果保存到 bottomA
rightA = leftA + A(:,3);  % 详解: 赋值：计算表达式并保存到 rightA
topA = bottomA + A(:,4);  % 详解: 赋值：计算表达式并保存到 topA

leftB = B(:,1)';  % 赋值：设置变量 leftB  % 详解: 赋值：将 B(...) 的结果保存到 leftB  % 详解: 赋值：将 B(...) 的结果保存到 leftB
bottomB = B(:,2)';  % 赋值：设置变量 bottomB  % 详解: 赋值：将 B(...) 的结果保存到 bottomB  % 详解: 赋值：将 B(...) 的结果保存到 bottomB
rightB = leftB + B(:,3)';  % 赋值：设置变量 rightB  % 详解: 赋值：计算表达式并保存到 rightB  % 详解: 赋值：计算表达式并保存到 rightB
topB = bottomB + B(:,4)';  % 赋值：设置变量 topB  % 详解: 赋值：计算表达式并保存到 topB  % 详解: 赋值：计算表达式并保存到 topB

numRectA = size(A,1);  % 详解: 赋值：将 size(...) 的结果保存到 numRectA
numRectB = size(B,1);  % 详解: 赋值：将 size(...) 的结果保存到 numRectB

verbose = 0;  % 详解: 赋值：计算表达式并保存到 verbose
[overlap, normoverlap] = rectintSparseLoopC(leftA, rightA, topA, bottomA, leftB, rightB, topB, bottomB, verbose);  % 详解: 执行语句




