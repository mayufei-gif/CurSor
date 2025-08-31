% 文件: optimalMatching.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% MATCH - Solves the weighted bipartite matching (or assignment)
%         problem.
%
% Usage:  a = match(C);
%
% Arguments:   
%         C     - an m x n cost matrix; the sets are taken to be
%                 1:m and 1:n; C(i, j) gives the cost of matching
%                 items i (of the first set) and j (of the second set)
%
% Returns:
%
%         a     - an m x 1 assignment vector, which gives the
%                 minimum cost assignment.  a(i) is the index of
%                 the item of 1:n that was matched to item i of
%                 1:m.  If item i (of 1:m) was not matched to any 
%                 item of 1:n, then a(i) is zero.

% Copyright (C) 2002 Mark A. Paskin
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by  % 中文: 根据|||发布的GNU通用公共许可条款的条款自由软件基金会；许可证的第2版，或||| （您可以选择）任何以后的版本。 |||该程序的分布是希望它将有用的，但是|||没有任何保修；甚至没有|||的隐含保证适合或适合特定目的的健身。  请参阅gnu |||通用公共许可证以获取更多详细信息。 |||您应该已经收到了GNU通用公共许可证的副本|||以及这个程序；如果没有，请写入免费软件||| Foundation，Inc。，59 Temple Place，Suite 330，马萨诸塞州波士顿02111-1307 |||美国。 |||琐碎的情况：|||首先，通过简单的最佳匹配来减少问题。  如果两个|||元素同意它们是最好的匹配，然后将它们匹配。 |||获取两组的（新）大小，u和v。||| mx = realmax; |||将亲和力矩阵为正方形|||运行匈牙利方法。  首先用|||替换无限值最大（或最小）有限值。
% the Free Software Foundation; either version 2 of the License, or  % 中文: fprintf（'跑步匈牙利\ n'）;
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
% USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [a] = optimalMatching(C)  % 详解: 函数定义：optimalMatching(C), 返回：a

[p, q] = size(C);  % 详解: 获取向量/矩阵尺寸
if (p == 0)  % 详解: 条件判断：if ((p == 0))
  a = [];  % 详解: 赋值：计算表达式并保存到 a
  return;  % 详解: 返回：从当前函数返回
elseif (q == 0)  % 详解: 条件判断：elseif ((q == 0))
  a = zeros(p, 1);  % 详解: 赋值：将 zeros(...) 的结果保存到 a
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句


if  0  % 详解: 条件判断：if (0)
[x, a] = min(C, [], 2);  % 详解: 统计：最大/最小值
[y, b] = min(C, [], 1);  % 详解: 统计：最大/最小值
u = find(1:p ~= b(a(:)));  % 详解: 赋值：将 find(...) 的结果保存到 u
a(u) = 0;  % 详解: 执行语句
v = find(1:q ~= a(b(:))');  % 赋值：设置变量 v  % 详解: 赋值：将 find(...) 的结果保存到 v  % 详解: 赋值：将 find(...) 的结果保存到 v
C = C(u, v);  % 详解: 赋值：将 C(...) 的结果保存到 C
if (isempty(C)) return; end  % 详解: 条件判断：if ((isempty(C)) return; end)
end  % 详解: 执行语句

[m, n] = size(C);  % 详解: 获取向量/矩阵尺寸

mx = 2*max(C(:));  % 详解: 赋值：计算表达式并保存到 mx
mn = -2*min(C(:));  % 详解: 赋值：计算表达式并保存到 mn
if (m < n)  % 详解: 条件判断：if ((m < n))
  C = [C; mx * ones(n - m, n)];  % 详解: 赋值：计算表达式并保存到 C
elseif (n < m)  % 详解: 条件判断：elseif ((n < m))
  C = [C, mx * ones(m, m - n)];  % 详解: 赋值：计算表达式并保存到 C
end  % 详解: 执行语句

C(find(isinf(C) & (C > 0))) = mx;  % 详解: 执行语句
C(find(isinf(C) & (C < 0))) = mn;  % 详解: 执行语句
[b, cost] = hungarian(C');  % 执行语句  % 详解: 执行语句  % 详解: 执行语句

ap = b(1:m)';  % 赋值：设置变量 ap  % 详解: 赋值：将 b(...) 的结果保存到 ap  % 详解: 赋值：将 b(...) 的结果保存到 ap
ap(find(ap > n)) = 0;  % 详解: 执行语句

a = ap;  % 详解: 赋值：计算表达式并保存到 a





