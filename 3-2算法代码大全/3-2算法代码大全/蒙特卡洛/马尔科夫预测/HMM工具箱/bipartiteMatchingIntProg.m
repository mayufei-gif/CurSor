% 文件: bipartiteMatchingIntProg.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [a,ass] = bipartiteMatchingIntProg(dst, nmatches)  % 详解: 函数定义：bipartiteMatchingIntProg(dst, nmatches), 返回：a,ass

if nargin < 2, nmatches = []; end  % 详解: 条件判断：if (nargin < 2, nmatches = []; end)

[p1 p2] = size(dst);  % 详解: 获取向量/矩阵尺寸
p1orig = p1; p2orig = p2;  % 详解: 赋值：计算表达式并保存到 p1orig
dstorig = dst;  % 详解: 赋值：计算表达式并保存到 dstorig

if isempty(nmatches)  % 详解: 条件判断：if (isempty(nmatches))
  m = max(dst(:));  % 详解: 赋值：将 max(...) 的结果保存到 m
  if p1<p2  % 详解: 条件判断：if (p1<p2)
    dst = [dst; m*ones(p2-p1, p2)];  % 详解: 赋值：计算表达式并保存到 dst
  elseif p1>p2  % 详解: 条件判断：elseif (p1>p2)
    dst = [dst  m*ones(p1, p1-p2)];  % 详解: 赋值：计算表达式并保存到 dst
  end  % 详解: 执行语句
end  % 详解: 执行语句
[p1 p2] = size(dst);  % 详解: 获取向量/矩阵尺寸


c = dst(:);  % 详解: 赋值：将 dst(...) 的结果保存到 c

A2 = kron(eye(p2), ones(1,p1));  % 详解: 赋值：将 kron(...) 的结果保存到 A2
b2 = ones(p2,1);  % 详解: 赋值：将 ones(...) 的结果保存到 b2

A3 = kron(ones(1,p2), eye(p1));  % 详解: 赋值：将 kron(...) 的结果保存到 A3
b3 = ones(p1,1);  % 详解: 赋值：将 ones(...) 的结果保存到 b3

if isempty(nmatches)  % 详解: 条件判断：if (isempty(nmatches))
  A = [A2; A3];  % 详解: 赋值：计算表达式并保存到 A
  b = [b2; b3];  % 详解: 赋值：计算表达式并保存到 b
  Aineq = zeros(1, p1*p2);  % 详解: 赋值：将 zeros(...) 的结果保存到 Aineq
  bineq = 0;  % 详解: 赋值：计算表达式并保存到 bineq
else  % 详解: 条件判断：else 分支
  nmatches = min([nmatches, p1, p2]);  % 详解: 赋值：将 min(...) 的结果保存到 nmatches
  Aineq = [A2; A3];  % 详解: 赋值：计算表达式并保存到 Aineq
  bineq = [b2; b3];  % 详解: 赋值：计算表达式并保存到 bineq
  A = ones(1,p1*p2);  % 详解: 赋值：将 ones(...) 的结果保存到 A
  b = nmatches;  % 详解: 赋值：计算表达式并保存到 b
end  % 详解: 执行语句


ass = bintprog(c, Aineq, bineq, A, b);  % 详解: 赋值：将 bintprog(...) 的结果保存到 ass
ass = reshape(ass, p1, p2);  % 详解: 赋值：将 reshape(...) 的结果保存到 ass

a = zeros(1, p1orig);  % 详解: 赋值：将 zeros(...) 的结果保存到 a
for i=1:p1orig  % 详解: for 循环：迭代变量 i 遍历 1:p1orig
  ndx = find(ass(i,:)==1);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
  if ~isempty(ndx) & (ndx <= p2orig)  % 详解: 条件判断：if (~isempty(ndx) & (ndx <= p2orig))
    a(i) = ndx;  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句






