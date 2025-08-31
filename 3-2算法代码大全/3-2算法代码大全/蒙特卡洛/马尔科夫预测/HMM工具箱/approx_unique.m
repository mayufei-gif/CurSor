% 文件: approx_unique.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [B, keep] = approx_unique(A, thresh, flag)  % 详解: 函数定义：approx_unique(A, thresh, flag), 返回：B, keep

keep = [];  % 详解: 赋值：计算表达式并保存到 keep

if nargin < 3 | isempty(flag)  % 详解: 条件判断：if (nargin < 3 | isempty(flag))
  A = sort(A)  % 详解: 赋值：将 sort(...) 的结果保存到 A
  B = A(1);  % 详解: 赋值：将 A(...) 的结果保存到 B
  for i=2:length(A)  % 详解: for 循环：迭代变量 i 遍历 2:length(A)
    if ~approxeq(A(i), A(i-1), thresh)  % 详解: 条件判断：if (~approxeq(A(i), A(i-1), thresh))
      B = [B A(i)];  % 详解: 赋值：计算表达式并保存到 B
      keep = [keep i];  % 详解: 赋值：计算表达式并保存到 keep
    end  % 详解: 执行语句
  end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  B = [];  % 详解: 赋值：计算表达式并保存到 B
  for i=1:size(A,1)  % 详解: for 循环：迭代变量 i 遍历 1:size(A,1)
    duplicate = 0;  % 详解: 赋值：计算表达式并保存到 duplicate
    for j=i+1:size(A,1)  % 详解: for 循环：迭代变量 j 遍历 i+1:size(A,1)
      if approxeq(A(i,:), A(j,:), thresh)  % 详解: 条件判断：if (approxeq(A(i,:), A(j,:), thresh))
	duplicate = 1;  % 详解: 赋值：计算表达式并保存到 duplicate
	break;  % 详解: 跳出循环：break
      end  % 详解: 执行语句
    end  % 详解: 执行语句
    if ~duplicate  % 详解: 条件判断：if (~duplicate)
      B = [B; A(i,:)];  % 详解: 赋值：计算表达式并保存到 B
      keep = [keep i];  % 详解: 赋值：计算表达式并保存到 keep
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句





