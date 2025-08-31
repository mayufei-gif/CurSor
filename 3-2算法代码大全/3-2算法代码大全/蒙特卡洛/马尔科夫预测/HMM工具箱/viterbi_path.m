% 文件: viterbi_path.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function path = viterbi_path(prior, transmat, obslik)  % 详解: 执行语句



scaled = 1;  % 详解: 赋值：计算表达式并保存到 scaled

T = size(obslik, 2);  % 详解: 赋值：将 size(...) 的结果保存到 T
prior = prior(:);  % 详解: 赋值：将 prior(...) 的结果保存到 prior
Q = length(prior);  % 详解: 赋值：将 length(...) 的结果保存到 Q

delta = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 delta
psi = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 psi
path = zeros(1,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 path
scale = ones(1,T);  % 详解: 赋值：将 ones(...) 的结果保存到 scale


t=1;  % 详解: 赋值：计算表达式并保存到 t
delta(:,t) = prior .* obslik(:,t);  % 详解: 调用函数：delta(:,t) = prior .* obslik(:,t)
if scaled  % 详解: 条件判断：if (scaled)
  [delta(:,t), n] = normalise(delta(:,t));  % 详解: 执行语句
  scale(t) = 1/n;  % 详解: 执行语句
end  % 详解: 执行语句
psi(:,t) = 0;  % 详解: 执行语句
for t=2:T  % 详解: for 循环：迭代变量 t 遍历 2:T
  for j=1:Q  % 详解: for 循环：迭代变量 j 遍历 1:Q
    [delta(j,t), psi(j,t)] = max(delta(:,t-1) .* transmat(:,j));  % 详解: 统计：最大/最小值
    delta(j,t) = delta(j,t) * obslik(j,t);  % 详解: 调用函数：delta(j,t) = delta(j,t) * obslik(j,t)
  end  % 详解: 执行语句
  if scaled  % 详解: 条件判断：if (scaled)
    [delta(:,t), n] = normalise(delta(:,t));  % 详解: 执行语句
    scale(t) = 1/n;  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句
[p, path(T)] = max(delta(:,T));  % 详解: 统计：最大/最小值
for t=T-1:-1:1  % 详解: for 循环：迭代变量 t 遍历 T-1:-1:1
  path(t) = psi(path(t+1),t+1);  % 详解: 调用函数：path(t) = psi(path(t+1),t+1)
end  % 详解: 执行语句


if 0  % 详解: 条件判断：if (0)
if scaled  % 详解: 条件判断：if (scaled)
  loglik = -sum(log(scale));  % 详解: 赋值：计算表达式并保存到 loglik
else  % 详解: 条件判断：else 分支
  loglik = log(p);  % 详解: 赋值：将 log(...) 的结果保存到 loglik
end  % 详解: 执行语句
end  % 详解: 执行语句




