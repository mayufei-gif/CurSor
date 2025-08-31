% 文件: fwdprop_backsample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function samples = fwdprop_backsample(init_state_distrib, transmat, obslik, nsamples)  % 详解: 执行语句



[Q T] = size(obslik);  % 详解: 获取向量/矩阵尺寸
scale = ones(1,T);  % 详解: 赋值：将 ones(...) 的结果保存到 scale
loglik = 0;  % 详解: 赋值：计算表达式并保存到 loglik
alpha = zeros(Q,T, 'single');  % 详解: 赋值：将 zeros(...) 的结果保存到 alpha
beta = zeros(Q,T,'single');  % 详解: 赋值：将 zeros(...) 的结果保存到 beta
gamma = zeros(Q,T,'single');  % 详解: 赋值：将 zeros(...) 的结果保存到 gamma
trans = transmat;  % 详解: 赋值：计算表达式并保存到 trans

t = 1;  % 详解: 赋值：计算表达式并保存到 t
alpha(:,1) = init_state_distrib(:) .* obslik(:,t);  % 详解: 调用函数：alpha(:,1) = init_state_distrib(:) .* obslik(:,t)
[alpha(:,t), scale(t)] = normalise(alpha(:,t));  % 详解: 执行语句
for t=2:T  % 详解: for 循环：迭代变量 t 遍历 2:T
  m = trans' * alpha(:,t-1);  % 赋值：设置变量 m  % 详解: 赋值：计算表达式并保存到 m  % 详解: 赋值：计算表达式并保存到 m
  alpha(:,t) = m(:) .* obslik(:,t);  % 详解: 调用函数：alpha(:,t) = m(:) .* obslik(:,t)
  [alpha(:,t), scale(t)] = normalise(alpha(:,t));  % 详解: 执行语句
  assert(approxeq(sum(alpha(:,t)),1))  % 详解: 调用函数：assert(approxeq(sum(alpha(:,t)),1))
end  % 详解: 执行语句
loglik = sum(log(scale));  % 详解: 赋值：将 sum(...) 的结果保存到 loglik


beta = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 beta
t=T;  % 详解: 赋值：计算表达式并保存到 t
beta(:,T) = ones(Q,1);  % 详解: 调用函数：beta(:,T) = ones(Q,1)
gamma(:,T) = normalize(alpha(:,T) .* beta(:,T));  % 详解: 调用函数：gamma(:,T) = normalize(alpha(:,T) .* beta(:,T))
if nsamples > 0  % 详解: 条件判断：if (nsamples > 0)
  samples(t,:) = sample(gamma(:,T), nsamples);  % 详解: 调用函数：samples(t,:) = sample(gamma(:,T), nsamples)
end  % 详解: 执行语句
for t=T-1:-1:1  % 详解: for 循环：迭代变量 t 遍历 T-1:-1:1
 b = beta(:,t+1) .* obslik(:,t+1);  % 详解: 赋值：将 beta(...) 的结果保存到 b
 beta(:,t) = normalize(transmat * b);  % 详解: 调用函数：beta(:,t) = normalize(transmat * b)
 gamma(:,t) = normalize(alpha(:,t) .* beta(:,t));  % 详解: 调用函数：gamma(:,t) = normalize(alpha(:,t) .* beta(:,t))
 if nsamples > 0  % 详解: 条件判断：if (nsamples > 0)
   xi_filtered = normalize((alpha(:,t) * obslik(:,t+1)') .* transmat);  % 赋值：设置变量 xi_filtered  % 详解: 赋值：将 normalize(...) 的结果保存到 xi_filtered  % 详解: 赋值：将 normalize(...) 的结果保存到 xi_filtered
   for n=1:nsamples  % 详解: for 循环：迭代变量 n 遍历 1:nsamples
     dist = normalize(xi_filtered(:,samples(t+1,n)));  % 详解: 赋值：将 normalize(...) 的结果保存到 dist
     samples(t,n) = sample(dist);  % 详解: 调用函数：samples(t,n) = sample(dist)
   end  % 详解: 执行语句
 end  % 详解: 执行语句
end  % 详解: 执行语句


if 0  % 详解: 条件判断：if (0)
beta(:,T) = ones(Q,1);  % 详解: 调用函数：beta(:,T) = ones(Q,1)
gamma(:,T) = normalise(alpha(:,T) .* beta(:,T));  % 详解: 调用函数：gamma(:,T) = normalise(alpha(:,T) .* beta(:,T))
t=T;  % 详解: 赋值：计算表达式并保存到 t
samples = zeros(T, nsamples);  % 详解: 赋值：将 zeros(...) 的结果保存到 samples
samples(t,:) = sample_discrete(gamma(:,t), 1, nsamples);  % 详解: 调用函数：samples(t,:) = sample_discrete(gamma(:,t), 1, nsamples)
for s=1:nsamples  % 详解: for 循环：迭代变量 s 遍历 1:nsamples
  for t=T-1:-1:1  % 详解: for 循环：迭代变量 t 遍历 T-1:-1:1
    L = samples(t+1,s);  % 详解: 赋值：将 samples(...) 的结果保存到 L
    obslikTmp = zeros(Q,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 obslikTmp
    obslikTmp(L) = 1;  % 详解: 执行语句
    b = beta(:,t+1) .* obslikTmp;  % 详解: 赋值：将 beta(...) 的结果保存到 b
    beta(:,t) = normalise(trans * b);  % 详解: 调用函数：beta(:,t) = normalise(trans * b)
    gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));  % 详解: 调用函数：gamma(:,t) = normalise(alpha(:,t) .* beta(:,t))
    samples(t,s) = sample_discrete(gamma(:,t), 1, 1);  % 详解: 调用函数：samples(t,s) = sample_discrete(gamma(:,t), 1, 1)
  end  % 详解: 执行语句
end  % 详解: 执行语句
end  % 详解: 执行语句




