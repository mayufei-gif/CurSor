% 文件: mixgauss_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [data, indices] = mixgauss_sample(mu, Sigma, mixweights, Nsamples)  % 详解: 函数定义：mixgauss_sample(mu, Sigma, mixweights, Nsamples), 返回：data, indices

[D K] = size(mu);  % 详解: 获取向量/矩阵尺寸
data = zeros(D, Nsamples);  % 详解: 赋值：将 zeros(...) 的结果保存到 data
indices = sample_discrete(mixweights, 1, Nsamples);  % 详解: 赋值：将 sample_discrete(...) 的结果保存到 indices
for k=1:K  % 详解: for 循环：迭代变量 k 遍历 1:K
  if ndims(Sigma) < 3  % 详解: 条件判断：if (ndims(Sigma) < 3)
    sig = Sigma(k);  % 详解: 赋值：将 Sigma(...) 的结果保存到 sig
  else  % 详解: 条件判断：else 分支
    sig = Sigma(:,:,k);  % 详解: 赋值：将 Sigma(...) 的结果保存到 sig
  end  % 详解: 执行语句
  ndx = find(indices==k);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
  if length(ndx) > 0  % 详解: 条件判断：if (length(ndx) > 0)
    data(:,ndx) = sample_gaussian(mu(:,k), sig, length(ndx))';  % 获取向量/矩阵尺寸信息  % 详解: 获取向量/矩阵尺寸  % 详解: 获取向量/矩阵尺寸
  end  % 详解: 执行语句
end  % 详解: 执行语句




