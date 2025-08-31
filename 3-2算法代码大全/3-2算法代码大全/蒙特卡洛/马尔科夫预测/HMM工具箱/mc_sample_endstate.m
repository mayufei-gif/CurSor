% 文件: mc_sample_endstate.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function S = sample_mc_endstate(startprob, trans, endprob)  % 详解: 执行语句

Q = size(trans,1);  % 详解: 赋值：将 size(...) 的结果保存到 Q
transprob = zeros(Q,Q+1);  % 详解: 赋值：将 zeros(...) 的结果保存到 transprob
end_state = Q+1;  % 详解: 赋值：计算表达式并保存到 end_state
for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
  for j=1:Q  % 详解: for 循环：迭代变量 j 遍历 1:Q
    transprob(i,j) = (1-endprob(i)) * trans(i,j);  % 详解: 调用函数：transprob(i,j) = (1-endprob(i)) * trans(i,j)
  end  % 详解: 执行语句
  transprob(i,end_state) = endprob(i);  % 详解: 调用函数：transprob(i,end_state) = endprob(i)
end  % 详解: 执行语句

S = [];  % 详解: 赋值：计算表达式并保存到 S
S(1) = sample_discrete(startprob);  % 详解: 调用函数：S(1) = sample_discrete(startprob)
t = 1;  % 详解: 赋值：计算表达式并保存到 t
p = endprob(S(t));  % 详解: 赋值：将 endprob(...) 的结果保存到 p
stop = (S(1) == end_state);  % 详解: 赋值：计算表达式并保存到 stop
while ~stop  % 详解: while 循环：当 (~stop) 为真时迭代
  S(t+1) = sample_discrete(transprob(S(t),:));  % 详解: 调用函数：S(t+1) = sample_discrete(transprob(S(t),:))
  stop = (S(t+1) == end_state);  % 详解: 赋值：计算表达式并保存到 stop
  t = t + 1;  % 详解: 赋值：计算表达式并保存到 t
end  % 详解: 执行语句
S = S(1:end-1);  % 详解: 赋值：将 S(...) 的结果保存到 S




