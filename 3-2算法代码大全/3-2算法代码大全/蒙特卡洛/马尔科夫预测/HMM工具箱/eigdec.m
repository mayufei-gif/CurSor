% 文件: eigdec.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [evals, evec] = eigdec(x, N)  % 详解: 函数定义：eigdec(x, N), 返回：evals, evec


if nargout == 1  % 详解: 条件判断：if (nargout == 1)
   evals_only = logical(1);  % 详解: 赋值：将 logical(...) 的结果保存到 evals_only
else  % 详解: 条件判断：else 分支
   evals_only = logical(0);  % 详解: 赋值：将 logical(...) 的结果保存到 evals_only
end  % 详解: 执行语句

if N ~= round(N) | N < 1 | N > size(x, 2)  % 详解: 条件判断：if (N ~= round(N) | N < 1 | N > size(x, 2))
   error('Number of PCs must be integer, >0, < dim');  % 详解: 调用函数：error('Number of PCs must be integer, >0, < dim')
end  % 详解: 执行语句

if evals_only  % 详解: 条件判断：if (evals_only)
   temp_evals = eig(x);  % 详解: 赋值：将 eig(...) 的结果保存到 temp_evals
else  % 详解: 条件判断：else 分支
   if (N/size(x, 2)) > 0.04  % 详解: 条件判断：if ((N/size(x, 2)) > 0.04)
     fprintf('netlab pca: using eig\n');  % 详解: 调用函数：fprintf('netlab pca: using eig\n')
      [temp_evec, temp_evals] = eig(x);  % 详解: 执行语句
   else  % 详解: 条件判断：else 分支
      options.disp = 0;  % 详解: 赋值：计算表达式并保存到 options.disp
      fprintf('netlab pca: using eigs\n');  % 详解: 调用函数：fprintf('netlab pca: using eigs\n')
      [temp_evec, temp_evals] = eigs(x, N, 'LM', options);  % 详解: 执行语句
   end  % 详解: 执行语句
   temp_evals = diag(temp_evals);  % 详解: 赋值：将 diag(...) 的结果保存到 temp_evals
end  % 详解: 执行语句

[evals perm] = sort(-temp_evals);  % 详解: 执行语句
evals = -evals(1:N);  % 详解: 赋值：计算表达式并保存到 evals
if ~evals_only  % 详解: 条件判断：if (~evals_only)
  if evals == temp_evals(1:N)  % 详解: 条件判断：if (evals == temp_evals(1:N))
    evec = temp_evec(:, 1:N);  % 详解: 赋值：将 temp_evec(...) 的结果保存到 evec
    return  % 详解: 返回：从当前函数返回
  else  % 详解: 条件判断：else 分支
    fprintf('netlab pca: sorting evec\n');  % 详解: 调用函数：fprintf('netlab pca: sorting evec\n')
    for i=1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
      evec(:,i) = temp_evec(:,perm(i));  % 详解: 调用函数：evec(:,i) = temp_evec(:,perm(i))
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句




