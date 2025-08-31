% 文件: polygon_intersect.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [is,S] = isintpl(x1,y1,x2,y2)  % 详解: 函数定义：isintpl(x1,y1,x2,y2), 返回：is,S



if nargin==0, help isintpl, return, end  % 详解: 条件判断：if (nargin==0, help isintpl, return, end)
if nargin<4  % 详解: 条件判断：if (nargin<4)
  error(' Not enough input arguments ')  % 详解: 调用函数：error(' Not enough input arguments ')
end  % 详解: 执行语句

x1 = x1(:); y1 = y1(:);  % 详解: 赋值：将 x1(...) 的结果保存到 x1
x2 = x2(:); y2 = y2(:);  % 详解: 赋值：将 x2(...) 的结果保存到 x2
l1 = length(x1);  % 详解: 赋值：将 length(...) 的结果保存到 l1
l2 = length(x2);  % 详解: 赋值：将 length(...) 的结果保存到 l2
if length(y1)~=l1 | length(y2)~=l2  % 详解: 条件判断：if (length(y1)~=l1 | length(y2)~=l2)
  error('(X1,Y1) and (X2,Y2) must pair-wise have the same length.')  % 详解: 调用函数：error('(X1,Y1) and (X2,Y2) must pair-wise have the same length.')
end  % 详解: 执行语句

if l1<1 | l2<1, is = []; S = []; return, end  % 详解: 条件判断：if (l1<1 | l2<1, is = []; S = []; return, end)

lim1 = [min(x1) max(x1)]';  % 统计：最大/最小值  % 详解: 赋值：计算表达式并保存到 lim1  % 详解: 赋值：计算表达式并保存到 lim1
lim2 = [min(x2) max(x2)]';  % 统计：最大/最小值  % 详解: 赋值：计算表达式并保存到 lim2  % 详解: 赋值：计算表达式并保存到 lim2
isx = interval(lim1,lim2);  % 详解: 赋值：将 interval(...) 的结果保存到 isx
lim1 = [min(y1) max(y1)]';  % 统计：最大/最小值  % 详解: 赋值：计算表达式并保存到 lim1  % 详解: 赋值：计算表达式并保存到 lim1
lim2 = [min(y2) max(y2)]';  % 统计：最大/最小值  % 详解: 赋值：计算表达式并保存到 lim2  % 详解: 赋值：计算表达式并保存到 lim2
isy = interval(lim1,lim2);  % 详解: 赋值：将 interval(...) 的结果保存到 isy
is = isx & isy;  % 详解: 赋值：计算表达式并保存到 is
S = zeros(l2,l1);  % 详解: 赋值：将 zeros(...) 的结果保存到 S

if ~is, return, end  % 详解: 条件判断：if (~is, return, end)

[i11,i12] = meshgrid(1:l1,1:l2);  % 详解: 执行语句
[i21,i22] = meshgrid([2:l1 1],[2:l2 1]);  % 详解: 执行语句
i11 = i11(:); i12 = i12(:);  % 详解: 赋值：将 i11(...) 的结果保存到 i11
i21 = i21(:); i22 = i22(:);  % 详解: 赋值：将 i21(...) 的结果保存到 i21

S(:) = iscross([x1(i11) x1(i21)]',[y1(i11) y1(i21)]',...  % 详解: 执行语句
               [x2(i12) x2(i22)]',[y2(i12) y2(i22)]')';  % 执行语句  % 详解: 执行语句  % 详解: 执行语句

S = S';  % 赋值：设置变量 S  % 详解: 赋值：计算表达式并保存到 S  % 详解: 赋值：计算表达式并保存到 S
is  = any(any(S));  % 详解: 赋值：将 any(...) 的结果保存到 is





