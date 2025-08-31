% 文件: mdp_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function state = sample_mdp(prior, trans, act)  % 详解: 执行语句

len = length(act);  % 详解: 赋值：将 length(...) 的结果保存到 len
state = zeros(1,len);  % 详解: 赋值：将 zeros(...) 的结果保存到 state
state(1) = sample_discrete(prior);  % 详解: 调用函数：state(1) = sample_discrete(prior)
for t=2:len  % 详解: for 循环：迭代变量 t 遍历 2:len
  state(t) = sample_discrete(trans{act(t)}(state(t-1),:));  % 详解: 调用函数：state(t) = sample_discrete(trans{act(t)}(state(t-1),:))
end  % 详解: 执行语句




