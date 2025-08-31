% 文件: condgauss_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function x = mixgauss_sample(mu, Sigma, labels)  % 详解: 执行语句

T = length(labels);  % 详解: 赋值：将 length(...) 的结果保存到 T
[D Q] = size(mu);  % 详解: 获取向量/矩阵尺寸
x = zeros(D,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 x
for q=1:Q  % 详解: for 循环：迭代变量 q 遍历 1:Q
  ndx = find(labels==q);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
  x(:,ndx) = gaussian_sample(mu(:,q)', Sigma(:,:,q), length(ndx))';  % 详解: 获取向量/矩阵尺寸
end  % 详解: 执行语句




