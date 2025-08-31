% 文件: STD.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function y = std(x,flag,dim)  % 详解: 统计：标准差/方差


if nargin<2, flag = 0; end  % 详解: 条件判断：if (nargin<2, flag = 0; end)
if nargin<3,  % 详解: 条件判断：if (nargin<3,)
  dim = min(find(size(x)~=1));  % 详解: 赋值：将 min(...) 的结果保存到 dim
  if isempty(dim), dim = 1; end  % 详解: 条件判断：if (isempty(dim), dim = 1; end)
end  % 详解: 执行语句

if size(x,dim)==1, y = zeros(size(x)); return, end  % 详解: 条件判断：if (size(x,dim)==1, y = zeros(size(x)); return, end)

tile = ones(1,max(ndims(x),dim));  % 详解: 赋值：将 ones(...) 的结果保存到 tile
tile(dim) = size(x,dim);  % 详解: 调用函数：tile(dim) = size(x,dim)

xc = x - repmat(sum(x,dim)/size(x,dim),tile);  % 详解: 赋值：计算表达式并保存到 xc
if flag,  % 详解: 条件判断：if (flag,)
  y = sqrt(sum(conj(xc).*xc,dim)/size(x,dim));  % 详解: 赋值：将 sqrt(...) 的结果保存到 y
else  % 详解: 条件判断：else 分支
  y = sqrt(sum(conj(xc).*xc,dim)/(size(x,dim)-1));  % 详解: 赋值：将 sqrt(...) 的结果保存到 y
end  % 详解: 执行语句





