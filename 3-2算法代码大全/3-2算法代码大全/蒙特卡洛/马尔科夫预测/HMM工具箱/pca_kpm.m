% 文件: pca_kpm.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [pc_vec]=pca_kpm(features,N, method);  % 详解: 函数定义：pca_kpm(features,N, method), 返回：pc_vec

[d ncases] = size(features);  % 详解: 获取向量/矩阵尺寸
fm=features-repmat(mean(features,2), 1, ncases);  % 详解: 赋值：计算表达式并保存到 fm


if method==1  % 详解: 条件判断：if (method==1)
  fprintf('pca_kpm eigs\n');  % 详解: 调用函数：fprintf('pca_kpm eigs\n')
  options.disp = 0;  % 详解: 赋值：计算表达式并保存到 options.disp
  C = cov(fm'); % d x d matrix  % 详解: 赋值：将 cov(...) 的结果保存到 C  % 详解: 赋值：将 cov(...) 的结果保存到 C
  [pc_vec, evals] = eigs(C, N, 'LM', options);  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  fprintf('pca_kpm svds\n');  % 详解: 调用函数：fprintf('pca_kpm svds\n')
  [U,D,V] = svds(fm', N);  % 执行语句  % 详解: 执行语句  % 详解: 执行语句
  pc_vec = V;  % 详解: 赋值：计算表达式并保存到 pc_vec
end  % 详解: 执行语句





