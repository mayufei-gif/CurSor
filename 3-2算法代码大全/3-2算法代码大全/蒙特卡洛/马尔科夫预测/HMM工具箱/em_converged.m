% 文件: em_converged.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [converged, decrease] = em_converged(loglik, previous_loglik, threshold, check_increased)  % 详解: 函数定义：em_converged(loglik, previous_loglik, threshold, check_increased), 返回：converged, decrease

if nargin < 3, threshold = 1e-4; end  % 详解: 条件判断：if (nargin < 3, threshold = 1e-4; end)
if nargin < 4, check_increased = 1; end  % 详解: 条件判断：if (nargin < 4, check_increased = 1; end)

converged = 0;  % 详解: 赋值：计算表达式并保存到 converged
decrease = 0;  % 详解: 赋值：计算表达式并保存到 decrease

if check_increased  % 详解: 条件判断：if (check_increased)
  if loglik - previous_loglik < -1e-3  % 详解: 条件判断：if (loglik - previous_loglik < -1e-3)
    fprintf(1, '******likelihood decreased from %6.4f to %6.4f!\n', previous_loglik, loglik);  % 详解: 调用函数：fprintf(1, '******likelihood decreased from %6.4f to %6.4f!\n', previous_loglik, loglik)
    decrease = 1;  % 详解: 赋值：计算表达式并保存到 decrease
converged = 0;  % 详解: 赋值：计算表达式并保存到 converged
return;  % 详解: 返回：从当前函数返回
  end  % 详解: 执行语句
end  % 详解: 执行语句

delta_loglik = abs(loglik - previous_loglik);  % 详解: 赋值：将 abs(...) 的结果保存到 delta_loglik
avg_loglik = (abs(loglik) + abs(previous_loglik) + eps)/2;  % 详解: 赋值：计算表达式并保存到 avg_loglik
if (delta_loglik / avg_loglik) < threshold, converged = 1; end  % 详解: 条件判断：if ((delta_loglik / avg_loglik) < threshold, converged = 1; end)




