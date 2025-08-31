% 文件: plot_polygon.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function out=plot_polygon(p, args, close_loop)  % 详解: 执行语句


if nargin < 2, args = []; end  % 详解: 条件判断：if (nargin < 2, args = []; end)
if nargin < 3, close_loop = 0; end  % 详解: 条件判断：if (nargin < 3, close_loop = 0; end)

if close_loop  % 详解: 条件判断：if (close_loop)
  p = [p p(:,1)];  % 详解: 赋值：计算表达式并保存到 p
end  % 详解: 执行语句

if isempty(args)  % 详解: 条件判断：if (isempty(args))
   out=plot(p(1,:),p(2,:));  % 详解: 赋值：将 plot(...) 的结果保存到 out
else  % 详解: 条件判断：else 分支
   out=plot(p(1,:),p(2,:),args);  % 详解: 赋值：将 plot(...) 的结果保存到 out
end  % 详解: 执行语句




