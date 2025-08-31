% 文件: sort_evec.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [evec, evals] = sort_evec(temp_evec, temp_evals, N)  % 详解: 函数定义：sort_evec(temp_evec, temp_evals, N), 返回：evec, evals

if ~isvector(temp_evals)  % 详解: 条件判断：if (~isvector(temp_evals))
  temp_evals = diag(temp_evals);  % 详解: 赋值：将 diag(...) 的结果保存到 temp_evals
end  % 详解: 执行语句

[evals perm] = sort(-temp_evals);  % 详解: 执行语句
evals = -evals(1:N);  % 详解: 赋值：计算表达式并保存到 evals
if evals == temp_evals(1:N)  % 详解: 条件判断：if (evals == temp_evals(1:N))
  evec = temp_evec(:, 1:N);  % 详解: 赋值：将 temp_evec(...) 的结果保存到 evec
  return  % 详解: 返回：从当前函数返回
else  % 详解: 条件判断：else 分支
  fprintf('sorting evec\n');  % 详解: 调用函数：fprintf('sorting evec\n')
  for i=1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
    evec(:,i) = temp_evec(:,perm(i));  % 详解: 调用函数：evec(:,i) = temp_evec(:,perm(i))
  end  % 详解: 执行语句
end  % 详解: 执行语句





