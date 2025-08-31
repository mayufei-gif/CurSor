% 文件: cross_entropy.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function kl = cross_entropy(p, q, symmetric)  % 详解: 执行语句

tiny = exp(-700);  % 详解: 赋值：将 exp(...) 的结果保存到 tiny
if nargin < 3, symmetric = 0; end  % 详解: 条件判断：if (nargin < 3, symmetric = 0; end)
p = p(:);  % 详解: 赋值：将 p(...) 的结果保存到 p
q = q(:);  % 详解: 赋值：将 q(...) 的结果保存到 q
if symmetric  % 详解: 条件判断：if (symmetric)
  kl  = (sum(p .* log((p+tiny)./(q+tiny))) + sum(q .* log((q+tiny)./(p+tiny))))/2;  % 详解: 赋值：计算表达式并保存到 kl
else  % 详解: 条件判断：else 分支
  kl  = sum(p .* log((p+tiny)./(q+tiny)));  % 详解: 赋值：将 sum(...) 的结果保存到 kl
end  % 详解: 执行语句




