% 文件: partitionData.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function  varargout = partitionData(Ndata, varargin)  % 详解: 执行语句

Npartitions = length(varargin);  % 详解: 赋值：将 length(...) 的结果保存到 Npartitions
perm = randperm(Ndata);  % 详解: 赋值：将 randperm(...) 的结果保存到 perm
ndx = 1;  % 详解: 赋值：计算表达式并保存到 ndx
for i=1:Npartitions  % 详解: for 循环：迭代变量 i 遍历 1:Npartitions
  pc(i) = varargin{i};  % 详解: 执行语句
  Nbin(i) = fix(Ndata*pc(i));  % 详解: 调用函数：Nbin(i) = fix(Ndata*pc(i))
  low(i) = ndx;  % 详解: 执行语句
  if i==Npartitions  % 详解: 条件判断：if (i==Npartitions)
    high(i) = Ndata;  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    high(i) = low(i)+Nbin(i)-1;  % 详解: 执行语句
  end  % 详解: 执行语句
  varargout{i} = perm(low(i):high(i));  % 详解: 执行语句
  ndx = ndx+Nbin(i);  % 详解: 赋值：计算表达式并保存到 ndx
end  % 详解: 执行语句





