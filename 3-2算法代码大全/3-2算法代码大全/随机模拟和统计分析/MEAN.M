% 文件: MEAN.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function y = mean(x,dim)  % 详解: 统计：求和/均值/中位数


if nargin==1,  % 详解: 条件判断：if (nargin==1,)
  dim = min(find(size(x)~=1));  % 详解: 赋值：将 min(...) 的结果保存到 dim
  if isempty(dim), dim = 1; end  % 详解: 条件判断：if (isempty(dim), dim = 1; end)

  y = sum(x)/size(x,dim);  % 详解: 赋值：将 sum(...) 的结果保存到 y
else  % 详解: 条件判断：else 分支
  y = sum(x,dim)/size(x,dim);  % 详解: 赋值：将 sum(...) 的结果保存到 y
end  % 详解: 执行语句




