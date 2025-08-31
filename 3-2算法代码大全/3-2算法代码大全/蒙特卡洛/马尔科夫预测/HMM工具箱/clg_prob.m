% 文件: clg_prob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p = eval_pdf_clg(X,Y,mu,Sigma,W)  % 详解: 执行语句

[d T] = size(Y);  % 详解: 获取向量/矩阵尺寸
[d nc] = size(mu);  % 详解: 获取向量/矩阵尺寸
p = zeros(nc,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 p
for c=1:nc  % 详解: for 循环：迭代变量 c 遍历 1:nc
  denom = (2*pi)^(d/2)*sqrt(abs(det(Sigma(:,:,c))));  % 详解: 赋值：计算表达式并保存到 denom
  M = repmat(mu(:,c), 1, T) + W(:,:,c)*X;  % 详解: 赋值：将 repmat(...) 的结果保存到 M
  mahal = sum(((Y-M)'*inv(Sigma(:,:,c))).*(Y-M)',2);  % 详解: 赋值：将 sum(...) 的结果保存到 mahal
  p(c,:) = (exp(-0.5*mahal) / denom)';  % 调用函数：p  % 详解: 执行语句  % 详解: 执行语句
end  % 详解: 执行语句




