% 文件: SORTROWS.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [y,ndx] = sortrows(x,col)  % 详解: 函数定义：sortrows(x,col), 返回：y,ndx


if nargin<1, error('Not enough input arguments.'); end  % 详解: 条件判断：if (nargin<1, error('Not enough input arguments.'); end)
if ndims(x)>2, error('X must be a 2-D matrix.'); end  % 详解: 条件判断：if (ndims(x)>2, error('X must be a 2-D matrix.'); end)

if nargin<2, col = 1:size(x,2); end  % 详解: 条件判断：if (nargin<2, col = 1:size(x,2); end)
if isempty(x), y = x; ndx = []; return, end  % 详解: 条件判断：if (isempty(x), y = x; ndx = []; return, end)

ndx = (1:size(x,1))';  % 获取向量/矩阵尺寸信息  % 详解: 赋值：计算表达式并保存到 ndx  % 详解: 赋值：计算表达式并保存到 ndx
for i=length(col):-1:1,  % 详解: for 循环：迭代变量 i 遍历 length(col):-1:1,
  [v,ind] = sort(x(ndx,col(i)));  % 详解: 执行语句
  ndx = ndx(ind);  % 详解: 赋值：将 ndx(...) 的结果保存到 ndx
end  % 详解: 执行语句
y = x(ndx,:);  % 详解: 赋值：将 x(...) 的结果保存到 y




