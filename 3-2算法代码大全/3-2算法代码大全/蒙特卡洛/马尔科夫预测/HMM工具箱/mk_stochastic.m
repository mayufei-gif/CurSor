% 文件: mk_stochastic.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [T,Z] = mk_stochastic(T)  % 详解: 函数定义：mk_stochastic(T), 返回：T,Z


if (ndims(T)==2) & (size(T,1)==1 | size(T,2)==1)  % 详解: 条件判断：if ((ndims(T)==2) & (size(T,1)==1 | size(T,2)==1))
  [T,Z] = normalise(T);  % 详解: 执行语句
elseif ndims(T)==2  % 详解: 条件判断：elseif (ndims(T)==2)
  Z = sum(T,2);  % 详解: 赋值：将 sum(...) 的结果保存到 Z
  S = Z + (Z==0);  % 详解: 赋值：计算表达式并保存到 S
  norm = repmat(S, 1, size(T,2));  % 详解: 赋值：将 repmat(...) 的结果保存到 norm
  T = T ./ norm;  % 详解: 赋值：计算表达式并保存到 T
else  % 详解: 条件判断：else 分支
  ns = size(T);  % 详解: 赋值：将 size(...) 的结果保存到 ns
  T = reshape(T, prod(ns(1:end-1)), ns(end));  % 详解: 赋值：将 reshape(...) 的结果保存到 T
  Z = sum(T,2);  % 详解: 赋值：将 sum(...) 的结果保存到 Z
  S = Z + (Z==0);  % 详解: 赋值：计算表达式并保存到 S
  norm = repmat(S, 1, ns(end));  % 详解: 赋值：将 repmat(...) 的结果保存到 norm
  T = T ./ norm;  % 详解: 赋值：计算表达式并保存到 T
  T = reshape(T, ns);  % 详解: 赋值：将 reshape(...) 的结果保存到 T
end  % 详解: 执行语句




