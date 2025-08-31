% 文件: collapse_mog.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [new_mu, new_Sigma, new_Sigma2] = collapse_mog(mu, Sigma, coefs)  % 详解: 函数定义：collapse_mog(mu, Sigma, coefs), 返回：new_mu, new_Sigma, new_Sigma2


new_mu = sum(mu * diag(coefs), 2);  % 详解: 赋值：将 sum(...) 的结果保存到 new_mu

n = length(new_mu);  % 详解: 赋值：将 length(...) 的结果保存到 n
new_Sigma = zeros(n,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 new_Sigma
new_Sigma2 = zeros(n,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 new_Sigma2
for j=1:length(coefs)  % 详解: for 循环：迭代变量 j 遍历 1:length(coefs)
  m = mu(:,j) - new_mu;  % 详解: 赋值：将 mu(...) 的结果保存到 m
  new_Sigma = new_Sigma + coefs(j) * (Sigma(:,:,j) + m*m');  % 赋值：设置变量 new_Sigma  % 详解: 赋值：计算表达式并保存到 new_Sigma  % 详解: 赋值：计算表达式并保存到 new_Sigma
  new_Sigma2 = new_Sigma2 + coefs(j) * (Sigma(:,:,j) + mu(:,j)*mu(:,j)');  % 赋值：设置变量 new_Sigma2  % 详解: 赋值：计算表达式并保存到 new_Sigma2  % 详解: 赋值：计算表达式并保存到 new_Sigma2
end  % 详解: 执行语句




