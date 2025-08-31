% 文件: gaussian_prob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = gaussian_prob(x, m, C, use_log)  % 详解: 执行语句


if nargin < 4, use_log = 0; end  % 详解: 条件判断：if (nargin < 4, use_log = 0; end)

if length(m)==1  % 详解: 条件判断：if (length(m)==1)
  x = x(:)';  % 赋值：设置变量 x  % 详解: 赋值：将 x(...) 的结果保存到 x  % 详解: 赋值：将 x(...) 的结果保存到 x
end  % 详解: 执行语句
[d N] = size(x);  % 详解: 获取向量/矩阵尺寸
m = m(:);  % 详解: 赋值：将 m(...) 的结果保存到 m
M = m*ones(1,N);  % 详解: 赋值：计算表达式并保存到 M
denom = (2*pi)^(d/2)*sqrt(abs(det(C)));  % 详解: 赋值：计算表达式并保存到 denom
mahal = sum(((x-M)'*inv(C)).*(x-M)',2);  % 详解: 赋值：将 sum(...) 的结果保存到 mahal
if any(mahal<0)  % 详解: 条件判断：if (any(mahal<0))
  warning('mahal < 0 => C is not psd')  % 详解: 调用函数：warning('mahal < 0 => C is not psd')
end  % 详解: 执行语句
if use_log  % 详解: 条件判断：if (use_log)
  p = -0.5*mahal - log(denom);  % 详解: 赋值：计算表达式并保存到 p
else  % 详解: 条件判断：else 分支
  p = exp(-0.5*mahal) / (denom+eps);  % 详解: 赋值：将 exp(...) 的结果保存到 p
end  % 详解: 执行语句




