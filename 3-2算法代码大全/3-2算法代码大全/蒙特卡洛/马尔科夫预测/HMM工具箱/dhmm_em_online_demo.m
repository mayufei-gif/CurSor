% 文件: dhmm_em_online_demo.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% Example of online EM applied to a simple POMDP with fixed action seq

clear all  % 详解: 执行语句

rand('state', 1);  % 详解: 调用函数：rand('state', 1)
O = 2;  % 详解: 赋值：计算表达式并保存到 O
S = 2;  % 详解: 赋值：计算表达式并保存到 S
A = 2;  % 详解: 赋值：计算表达式并保存到 A
prior0 = [1 0]';  % 赋值：设置变量 prior0  % 详解: 赋值：计算表达式并保存到 prior0  % 详解: 赋值：计算表达式并保存到 prior0
transmat0 = cell(1,A);  % 详解: 赋值：将 cell(...) 的结果保存到 transmat0
transmat0{1} = [0.9 0.1; 0.1 0.9];  % 详解: 执行语句
transmat0{2} = [0.1 0.9; 0.9 0.1];  % 详解: 执行语句
obsmat0 = eye(2);  % 详解: 赋值：将 eye(...) 的结果保存到 obsmat0
	   

T = 10;  % 详解: 赋值：计算表达式并保存到 T
act = [1*ones(1,25) 2*ones(1,25) 1*ones(1,25) 2*ones(1,25)];  % 详解: 赋值：计算表达式并保存到 act
data = pomdp_sample(prior0, transmat0, obsmat0, act);  % 详解: 赋值：将 pomdp_sample(...) 的结果保存到 data

rand('state', 2);  % 详解: 调用函数：rand('state', 2)
transmat1 = cell(1,A);  % 详解: 赋值：将 cell(...) 的结果保存到 transmat1
for a=1:A  % 详解: for 循环：迭代变量 a 遍历 1:A
  transmat1{a} = mk_stochastic(rand(S,S));  % 详解: 生成随机数/矩阵
end  % 详解: 执行语句
obsmat1 = mk_stochastic(rand(S,O));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 obsmat1
prior1 = prior0;  % 详解: 赋值：计算表达式并保存到 prior1

e = 0.001;  % 详解: 赋值：计算表达式并保存到 e
ess_trans = cell(1,A);  % 详解: 赋值：将 cell(...) 的结果保存到 ess_trans
for a=1:A  % 详解: for 循环：迭代变量 a 遍历 1:A
  ess_trans{a} = repmat(e, S, S);  % 详解: 执行语句
end  % 详解: 执行语句
ess_emit = repmat(e, S, O);  % 详解: 赋值：将 repmat(...) 的结果保存到 ess_emit

w = 2;  % 详解: 赋值：计算表达式并保存到 w
decay_sched = [0.1:0.1:0.9];  % 详解: 赋值：计算表达式并保存到 decay_sched

LL1 = zeros(1,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 LL1
t = 1;  % 详解: 赋值：计算表达式并保存到 t
y = data(t);  % 详解: 赋值：将 data(...) 的结果保存到 y
data_win = y;  % 详解: 赋值：计算表达式并保存到 data_win
act_win = [1];  % 详解: 赋值：计算表达式并保存到 act_win
[prior1, LL1(1)] = normalise(prior1 .* obsmat1(:,y));  % 详解: 执行语句

for t=2:T  % 详解: for 循环：迭代变量 t 遍历 2:T
  y = data(t);  % 详解: 赋值：将 data(...) 的结果保存到 y
  a = act(t);  % 详解: 赋值：将 act(...) 的结果保存到 a
  if t <= w  % 详解: 条件判断：if (t <= w)
    data_win = [data_win y];  % 详解: 赋值：计算表达式并保存到 data_win
    act_win = [act_win a];  % 详解: 赋值：计算表达式并保存到 act_win
  else  % 详解: 条件判断：else 分支
    data_win = [data_win(2:end) y];  % 详解: 赋值：计算表达式并保存到 data_win
    act_win = [act_win(2:end) a];  % 详解: 赋值：计算表达式并保存到 act_win
    prior1 = gamma(:, 2);  % 详解: 赋值：将 gamma(...) 的结果保存到 prior1
  end  % 详解: 执行语句
  d = decay_sched(min(t, length(decay_sched)));  % 详解: 赋值：将 decay_sched(...) 的结果保存到 d
  [transmat1, obsmat1, ess_trans, ess_emit, gamma, ll] = dhmm_em_online(...  % 详解: 执行语句
      prior1, transmat1, obsmat1, ess_trans, ess_emit, d, data_win, act_win);  % 详解: 执行语句
  bel = gamma(:, end);  % 详解: 赋值：将 gamma(...) 的结果保存到 bel
  LL1(t) = ll/length(data_win);  % 详解: 调用函数：LL1(t) = ll/length(data_win)
end  % 详解: 执行语句

LL1(1) = LL1(2);  % 详解: 调用函数：LL1(1) = LL1(2)
plot(1:T, LL1, 'rx-');  % 详解: 调用函数：plot(1:T, LL1, 'rx-')



if 0  % 详解: 条件判断：if (0)
rand('state', 2);  % 详解: 调用函数：rand('state', 2)
transmat2 = cell(1,A);  % 详解: 赋值：将 cell(...) 的结果保存到 transmat2
for a=1:A  % 详解: for 循环：迭代变量 a 遍历 1:A
  transmat2{a} = mk_stochastic(rand(S,S));  % 详解: 生成随机数/矩阵
end  % 详解: 执行语句
obsmat2 = mk_stochastic(rand(S,O));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 obsmat2
prior2 = prior0;  % 详解: 赋值：计算表达式并保存到 prior2
[LL2, prior2, transmat2, obsmat2] = dhmm_em(data, prior2, transmat2, obsmat2, ....  % 详解: 执行语句
					       'max_iter', 10, 'thresh', 1e-3, 'verbose', 1, 'act', act);  % 详解: 执行语句

LL2 = LL2 / T  % 详解: 赋值：计算表达式并保存到 LL2

end  % 详解: 执行语句




