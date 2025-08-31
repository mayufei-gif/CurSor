% 文件: rectintSparse.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [overlap, normoverlap] = rectintSparse(A,B)  % 详解: 函数定义：rectintSparse(A,B), 返回：overlap, normoverlap

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


nnz = ceil(0.2*numRectA*numRectB);  % 详解: 赋值：将 ceil(...) 的结果保存到 nnz
overlap = sparse([], [], [], numRectA, numRectB, nnz);  % 详解: 赋值：将 sparse(...) 的结果保存到 overlap
normoverlap = sparse([], [], [], numRectA, numRectB, nnz);  % 详解: 赋值：将 sparse(...) 的结果保存到 normoverlap
for j=1:numRectB  % 详解: for 循环：迭代变量 j 遍历 1:numRectB
  for i=1:numRectA  % 详解: for 循环：迭代变量 i 遍历 1:numRectA
    tmp = (max(0, min(rightA(i), rightB(j)) - max(leftA(i), leftB(j)) ) ) .* ...  % 详解: 赋值：计算表达式并保存到 tmp
	(max(0, min(topA(i), topB(j)) - max(bottomA(i), bottomB(j)) ) );  % 详解: 统计：最大/最小值
    if tmp>0  % 详解: 条件判断：if (tmp>0)
      overlap(i,j) = tmp;  % 详解: 执行语句
      areaA = (rightA(i)-leftA(i))*(topA(i)-bottomA(i));  % 详解: 赋值：计算表达式并保存到 areaA
      areaB = (rightB(j)-leftB(j))*(topB(j)-bottomB(j));  % 详解: 赋值：计算表达式并保存到 areaB
      normoverlap(i,j) = min(tmp/areaA, tmp/areaB);  % 详解: 调用函数：normoverlap(i,j) = min(tmp/areaA, tmp/areaB)
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句


if 0  % 详解: 条件判断：if (0)
N = size(bboxDense01,2);  % 详解: 赋值：将 size(...) 的结果保存到 N
rect = bboxToRect(bboxDense01)';  % 赋值：设置变量 rect  % 详解: 赋值：将 bboxToRect(...) 的结果保存到 rect  % 详解: 赋值：将 bboxToRect(...) 的结果保存到 rect
A = rect(1:2,:);  % 详解: 赋值：将 rect(...) 的结果保存到 A
B = rect(1:N,:);  % 详解: 赋值：将 rect(...) 的结果保存到 B

tic; out1 = rectint(A, B); toc  % 详解: 执行语句
tic; out2 = rectintSparse(A, B); toc  % 详解: 执行语句
tic; out3 = rectintSparseC(A, B); toc  % 详解: 执行语句
tic; out4 = rectintC(A, B); toc  % 详解: 执行语句
assert(approxeq(out1, out2))  % 详解: 调用函数：assert(approxeq(out1, out2))
assert(approxeq(out1, full(out3)))  % 详解: 调用函数：assert(approxeq(out1, full(out3)))
assert(approxeq(out1, out4))  % 详解: 调用函数：assert(approxeq(out1, out4))
end  % 详解: 执行语句




