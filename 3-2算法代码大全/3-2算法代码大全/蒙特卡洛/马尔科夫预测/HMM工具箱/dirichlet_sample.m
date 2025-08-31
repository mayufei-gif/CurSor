% 文件: dirichlet_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function theta = dirichlet_sample(alpha, N)  % 详解: 执行语句


assert(alpha > 0);  % 详解: 调用函数：assert(alpha > 0)
k = length(alpha);  % 详解: 赋值：将 length(...) 的结果保存到 k
theta = zeros(N, k);  % 详解: 赋值：将 zeros(...) 的结果保存到 theta
scale = 1;  % 详解: 赋值：计算表达式并保存到 scale
for i=1:k  % 详解: for 循环：迭代变量 i 遍历 1:k
  theta(:,i) = gamma_sample(alpha(i), scale, N, 1);  % 详解: 调用函数：theta(:,i) = gamma_sample(alpha(i), scale, N, 1)
end  % 详解: 执行语句
S = sum(theta,2);  % 详解: 赋值：将 sum(...) 的结果保存到 S
theta = theta ./ repmat(S, 1, k);  % 详解: 赋值：计算表达式并保存到 theta




