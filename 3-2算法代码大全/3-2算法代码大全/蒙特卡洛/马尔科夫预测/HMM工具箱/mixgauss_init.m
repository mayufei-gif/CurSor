% 文件: mixgauss_init.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [mu, Sigma, weights] = mixgauss_init(M, data, cov_type, method)  % 详解: 函数定义：mixgauss_init(M, data, cov_type, method), 返回：mu, Sigma, weights

if nargin < 4, method = 'kmeans'; end  % 详解: 条件判断：if (nargin < 4, method = 'kmeans'; end)

[d T] = size(data);  % 详解: 获取向量/矩阵尺寸
data = reshape(data, d, T);  % 详解: 赋值：将 reshape(...) 的结果保存到 data

switch method  % 详解: 多分支选择：switch (method)
 case 'rnd',  % 详解: 分支：case 'rnd',
  C = cov(data');  % 赋值：设置变量 C  % 详解: 赋值：将 cov(...) 的结果保存到 C  % 详解: 赋值：将 cov(...) 的结果保存到 C
  Sigma = repmat(diag(diag(C))*0.5, [1 1 M]);  % 详解: 赋值：将 repmat(...) 的结果保存到 Sigma
  indices = randperm(T);  % 详解: 赋值：将 randperm(...) 的结果保存到 indices
  mu = data(:,indices(1:M));  % 详解: 赋值：将 data(...) 的结果保存到 mu
  weights = normalise(ones(M,1));  % 详解: 赋值：将 normalise(...) 的结果保存到 weights
 case 'kmeans',  % 详解: 分支：case 'kmeans',
  mix = gmm(d, M, cov_type);  % 详解: 赋值：将 gmm(...) 的结果保存到 mix
  options = foptions;  % 详解: 赋值：计算表达式并保存到 options
  max_iter = 5;  % 详解: 赋值：计算表达式并保存到 max_iter
  options(1) = -1;  % 详解: 执行语句
  options(14) = max_iter;  % 详解: 执行语句
  mix = gmminit(mix, data', options);  % 赋值：设置变量 mix  % 详解: 赋值：将 gmminit(...) 的结果保存到 mix  % 详解: 赋值：将 gmminit(...) 的结果保存到 mix
  mu = reshape(mix.centres', [d M]);  % 赋值：设置变量 mu  % 详解: 赋值：将 reshape(...) 的结果保存到 mu  % 详解: 赋值：将 reshape(...) 的结果保存到 mu
  weights = mix.priors(:);  % 详解: 赋值：计算表达式并保存到 weights
  for m=1:M  % 详解: for 循环：迭代变量 m 遍历 1:M
    switch cov_type  % 详解: 多分支选择：switch (cov_type)
     case 'diag',  % 详解: 分支：case 'diag',
      Sigma(:,:,m) = diag(mix.covars(m,:));  % 详解: 调用函数：Sigma(:,:,m) = diag(mix.covars(m,:))
     case 'full',  % 详解: 分支：case 'full',
      Sigma(:,:,m) = mix.covars(:,:,m);  % 详解: 调用函数：Sigma(:,:,m) = mix.covars(:,:,m)
     case 'spherical',  % 详解: 分支：case 'spherical',
      Sigma(:,:,m) = mix.covars(m) * eye(d);  % 详解: 调用函数：Sigma(:,:,m) = mix.covars(m) * eye(d)
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句





