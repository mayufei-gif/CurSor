% 文件: logsumexp.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function s = logsumexp(a, dim)  % 详解: 执行语句


if nargin < 2  % 详解: 条件判断：if (nargin < 2)
  dim = 1;  % 详解: 赋值：计算表达式并保存到 dim
  if ndims(a) <= 2 & size(a,1)==1  % 详解: 条件判断：if (ndims(a) <= 2 & size(a,1)==1)
    dim = 2;  % 详解: 赋值：计算表达式并保存到 dim
  end  % 详解: 执行语句
end  % 详解: 执行语句

[y, i] = max(a,[],dim);  % 详解: 统计：最大/最小值
dims = ones(1,ndims(a));  % 详解: 赋值：将 ones(...) 的结果保存到 dims
dims(dim) = size(a,dim);  % 详解: 调用函数：dims(dim) = size(a,dim)
a = a - repmat(y, dims);  % 详解: 赋值：计算表达式并保存到 a
s = y + log(sum(exp(a),dim));  % 详解: 赋值：计算表达式并保存到 s




