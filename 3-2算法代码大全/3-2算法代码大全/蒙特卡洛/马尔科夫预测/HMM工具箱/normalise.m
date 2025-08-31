% 文件: normalise.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [M, z] = normalise(A, dim)  % 详解: 函数定义：normalise(A, dim), 返回：M, z

if nargin < 2  % 详解: 条件判断：if (nargin < 2)
  z = sum(A(:));  % 详解: 赋值：将 sum(...) 的结果保存到 z
  s = z + (z==0);  % 详解: 赋值：计算表达式并保存到 s
  M = A / s;  % 详解: 赋值：计算表达式并保存到 M
elseif dim==1  % 详解: 条件判断：elseif (dim==1)
  z = sum(A);  % 详解: 赋值：将 sum(...) 的结果保存到 z
  s = z + (z==0);  % 详解: 赋值：计算表达式并保存到 s
  M = A ./ repmatC(s, size(A,1), 1);  % 详解: 赋值：计算表达式并保存到 M
else  % 详解: 条件判断：else 分支
  z=sum(A,dim);  % 详解: 赋值：将 sum(...) 的结果保存到 z
  s = z + (z==0);  % 详解: 赋值：计算表达式并保存到 s
  L=size(A,dim);  % 详解: 赋值：将 size(...) 的结果保存到 L
  d=length(size(A));  % 详解: 赋值：将 length(...) 的结果保存到 d
  v=ones(d,1);  % 详解: 赋值：将 ones(...) 的结果保存到 v
  v(dim)=L;  % 详解: 执行语句
  c=repmat(s,v');  % 赋值：设置变量 c  % 详解: 赋值：将 repmat(...) 的结果保存到 c  % 详解: 赋值：将 repmat(...) 的结果保存到 c
  M=A./c;  % 详解: 赋值：计算表达式并保存到 M
end  % 详解: 执行语句






