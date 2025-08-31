% 文件: fixed_lag_smoother_demo.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% Example of fixed lag smoothing

rand('state', 1);  % 详解: 调用函数：rand('state', 1)
S = 2;  % 详解: 赋值：计算表达式并保存到 S
O = 2;  % 详解: 赋值：计算表达式并保存到 O
T = 7;  % 详解: 赋值：计算表达式并保存到 T
data = sample_discrete([0.5 0.5], 1, T);  % 详解: 赋值：将 sample_discrete(...) 的结果保存到 data
transmat = mk_stochastic(rand(S,S));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat
obsmat = mk_stochastic(rand(S,O));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 obsmat
obslik = multinomial_prob(data, obsmat);  % 详解: 赋值：将 multinomial_prob(...) 的结果保存到 obslik
prior = [0.5 0.5]';  % 赋值：设置变量 prior  % 详解: 赋值：计算表达式并保存到 prior  % 详解: 赋值：计算表达式并保存到 prior


[alpha0, beta0, gamma0, ll0, xi0] = fwdback(prior, transmat, obslik);  % 详解: 执行语句

w = 3;  % 详解: 赋值：计算表达式并保存到 w
alpha1 = zeros(S, T);  % 详解: 赋值：将 zeros(...) 的结果保存到 alpha1
gamma1 = zeros(S, T);  % 详解: 赋值：将 zeros(...) 的结果保存到 gamma1
xi1 = zeros(S, S, T-1);  % 详解: 赋值：将 zeros(...) 的结果保存到 xi1
t = 1;  % 详解: 赋值：计算表达式并保存到 t
b = obsmat(:, data(t));  % 详解: 赋值：将 obsmat(...) 的结果保存到 b
olik_win = b;  % 详解: 赋值：计算表达式并保存到 olik_win
alpha_win = normalise(prior .* b);  % 详解: 赋值：将 normalise(...) 的结果保存到 alpha_win
alpha1(:,t) = alpha_win;  % 详解: 执行语句
for t=2:T  % 详解: for 循环：迭代变量 t 遍历 2:T
  [alpha_win, olik_win, gamma_win, xi_win] = ...  % 详解: 执行语句
      fixed_lag_smoother(w, alpha_win, olik_win, obsmat(:, data(t)), transmat);  % 详解: 调用函数：fixed_lag_smoother(w, alpha_win, olik_win, obsmat(:, data(t)), transmat)
  alpha1(:,max(1,t-w+1):t) = alpha_win;  % 详解: 统计：最大/最小值
  gamma1(:,max(1,t-w+1):t) = gamma_win;  % 详解: 统计：最大/最小值
  xi1(:,:,max(1,t-w+1):t-1) = xi_win;  % 详解: 统计：最大/最小值
end  % 详解: 执行语句

e = 1e-1;  % 详解: 赋值：计算表达式并保存到 e
assert(approxeq(gamma0(:, T-w+1:end), gamma1(:, T-w+1:end), e));  % 详解: 调用函数：assert(approxeq(gamma0(:, T-w+1:end), gamma1(:, T-w+1:end), e))






