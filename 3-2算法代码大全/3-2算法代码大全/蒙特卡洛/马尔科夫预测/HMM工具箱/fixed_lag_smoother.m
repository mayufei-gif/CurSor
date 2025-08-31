% 文件: fixed_lag_smoother.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [alpha, obslik, gamma, xi] = fixed_lag_smoother(d, alpha, obslik, obsvec, transmat, act)  % 详解: 函数定义：fixed_lag_smoother(d, alpha, obslik, obsvec, transmat, act), 返回：alpha, obslik, gamma, xi

[S n] = size(alpha);  % 详解: 获取向量/矩阵尺寸
d = min(d, n+1);  % 详解: 赋值：将 min(...) 的结果保存到 d
if d < 2  % 详解: 条件判断：if (d < 2)
  error('must keep a window of length at least 2');  % 详解: 调用函数：error('must keep a window of length at least 2')
end  % 详解: 执行语句

if ~exist('act')  % 详解: 条件判断：if (~exist('act'))
  act = ones(1, n+1);  % 详解: 赋值：将 ones(...) 的结果保存到 act
  transmat = { transmat };  % 详解: 赋值：计算表达式并保存到 transmat
end  % 详解: 执行语句

alpha = alpha(:, n-d+2:n);  % 详解: 赋值：将 alpha(...) 的结果保存到 alpha
obslik = obslik(:, n-d+2:n);  % 详解: 赋值：将 obslik(...) 的结果保存到 obslik

t = d;  % 详解: 赋值：计算表达式并保存到 t
obslik(:,t) = obsvec;  % 详解: 执行语句
xi = zeros(S, S, d-1);  % 详解: 赋值：将 zeros(...) 的结果保存到 xi
xi(:,:,t-1) = normalise((alpha(:,t-1) * obslik(:,t)') .* transmat{act(t)});  % 调用函数：xi  % 详解: 调用函数：xi(:,:,t-1) = normalise((alpha(:,t-1) * obslik(:,t)') .* transmat{act(t)})  % 详解: 调用函数：xi(:,:,t-1) = normalise((alpha(:,t-1) * obslik(:,t)') .* transmat{act(t)}); % 调用函数：xi % 详解: 调用函数：xi(:,:,t-1) = normalise((alpha(:,t-1) * obslik(:,t)') .* transmat{act(t)})
alpha(:,t) = sum(xi(:,:,t-1), 1)';  % 统计：求和/均值/中位数  % 详解: 统计：求和/均值/中位数  % 详解: 统计：求和/均值/中位数

beta = ones(S, d);  % 详解: 赋值：将 ones(...) 的结果保存到 beta
T = d;  % 详解: 赋值：计算表达式并保存到 T
gamma(:,T) = alpha(:,T);  % 详解: 调用函数：gamma(:,T) = alpha(:,T)
for t=T-1:-1:1  % 详解: for 循环：迭代变量 t 遍历 T-1:-1:1
  b = beta(:,t+1) .* obslik(:,t+1);  % 详解: 赋值：将 beta(...) 的结果保存到 b
  beta(:,t) = normalise(transmat{act(t)} * b);  % 详解: 调用函数：beta(:,t) = normalise(transmat{act(t)} * b)
  gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));  % 详解: 调用函数：gamma(:,t) = normalise(alpha(:,t) .* beta(:,t))
  xi(:,:,t) = normalise((transmat{act(t)} .* (alpha(:,t) * b')));  % 调用函数：xi  % 详解: 调用函数：xi(:,:,t) = normalise((transmat{act(t)} .* (alpha(:,t) * b')))  % 详解: 调用函数：xi(:,:,t) = normalise((transmat{act(t)} .* (alpha(:,t) * b'))); % 调用函数：xi % 详解: 调用函数：xi(:,:,t) = normalise((transmat{act(t)} .* (alpha(:,t) * b')))
end  % 详解: 执行语句








