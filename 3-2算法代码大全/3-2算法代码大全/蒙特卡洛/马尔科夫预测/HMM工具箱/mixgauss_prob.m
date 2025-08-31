% 文件: mixgauss_prob.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [B, B2] = mixgauss_prob(data, mu, Sigma, mixmat, unit_norm)  % 详解: 函数定义：mixgauss_prob(data, mu, Sigma, mixmat, unit_norm), 返回：B, B2




if isvector(mu) & size(mu,2)==1  % 详解: 条件判断：if (isvector(mu) & size(mu,2)==1)
  d = length(mu);  % 详解: 赋值：将 length(...) 的结果保存到 d
  Q = 1; M = 1;  % 详解: 赋值：计算表达式并保存到 Q
elseif ndims(mu)==2  % 详解: 条件判断：elseif (ndims(mu)==2)
  [d Q] = size(mu);  % 详解: 获取向量/矩阵尺寸
  M = 1;  % 详解: 赋值：计算表达式并保存到 M
else  % 详解: 条件判断：else 分支
  [d Q M] = size(mu);  % 详解: 获取向量/矩阵尺寸
end  % 详解: 执行语句
[d T] = size(data);  % 详解: 获取向量/矩阵尺寸

if nargin < 4, mixmat = ones(Q,1); end  % 详解: 条件判断：if (nargin < 4, mixmat = ones(Q,1); end)
if nargin < 5, unit_norm = 0; end  % 详解: 条件判断：if (nargin < 5, unit_norm = 0; end)


if isscalar(Sigma)  % 详解: 条件判断：if (isscalar(Sigma))
  mu = reshape(mu, [d Q*M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu
  if unit_norm  % 详解: 条件判断：if (unit_norm)
    disp('unit norm')  % 详解: 调用函数：disp('unit norm')
    D = 2 - 2*(mu'*data);  % 赋值：设置变量 D  % 详解: 赋值：计算表达式并保存到 D  % 详解: 赋值：计算表达式并保存到 D
    tic; D2 = sqdist(data, mu)'; toc  % 执行语句  % 详解: 执行语句  % 详解: 执行语句
    assert(approxeq(D,D2))  % 详解: 调用函数：assert(approxeq(D,D2))
  else  % 详解: 条件判断：else 分支
    D = sqdist(data, mu)';  % 赋值：设置变量 D  % 详解: 赋值：将 sqdist(...) 的结果保存到 D  % 详解: 赋值：将 sqdist(...) 的结果保存到 D
  end  % 详解: 执行语句
  clear mu data  % 详解: 执行语句
  logB2 = -(d/2)*log(2*pi*Sigma) - (1/(2*Sigma))*D;  % 详解: 赋值：计算表达式并保存到 logB2
  B2 = reshape(exp(logB2), [Q M T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 B2
  clear logB2  % 详解: 执行语句
  
elseif ndims(Sigma)==2  % 详解: 条件判断：elseif (ndims(Sigma)==2)
  mu = reshape(mu, [d Q*M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu
  D = sqdist(data, mu, inv(Sigma))';  % 矩阵求逆  % 详解: 赋值：将 sqdist(...) 的结果保存到 D  % 详解: 赋值：将 sqdist(...) 的结果保存到 D
  logB2 = -(d/2)*log(2*pi) - 0.5*logdet(Sigma) - 0.5*D;  % 详解: 赋值：计算表达式并保存到 logB2
  B2 = reshape(exp(logB2), [Q M T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 B2
  
elseif ndims(Sigma)==3  % 详解: 条件判断：elseif (ndims(Sigma)==3)
  B2 = zeros(Q,M,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 B2
  for j=1:Q  % 详解: for 循环：迭代变量 j 遍历 1:Q
    if isposdef(Sigma(:,:,j))  % 详解: 条件判断：if (isposdef(Sigma(:,:,j)))
      D = sqdist(data, permute(mu(:,j,:), [1 3 2]), inv(Sigma(:,:,j)))';  % 矩阵求逆  % 详解: 赋值：将 sqdist(...) 的结果保存到 D  % 详解: 赋值：将 sqdist(...) 的结果保存到 D
      logB2 = -(d/2)*log(2*pi) - 0.5*logdet(Sigma(:,:,j)) - 0.5*D;  % 详解: 赋值：计算表达式并保存到 logB2
      B2(j,:,:) = exp(logB2);  % 详解: 调用函数：B2(j,:,:) = exp(logB2)
    else  % 详解: 条件判断：else 分支
      error(sprintf('mixgauss_prob: Sigma(:,:,q=%d) not psd\n', j));  % 详解: 调用函数：error(sprintf('mixgauss_prob: Sigma(:,:,q=%d) not psd\n', j))
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  
else  % 详解: 条件判断：else 分支
  B2 = zeros(Q,M,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 B2
  for j=1:Q  % 详解: for 循环：迭代变量 j 遍历 1:Q
    for k=1:M  % 详解: for 循环：迭代变量 k 遍历 1:M
      B2(j,k,:) = gaussian_prob(data, mu(:,j,k), Sigma(:,:,j,k));  % 详解: 调用函数：B2(j,k,:) = gaussian_prob(data, mu(:,j,k), Sigma(:,:,j,k))
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句



  
B = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 B
if Q < T  % 详解: 条件判断：if (Q < T)
  for q=1:Q  % 详解: for 循环：迭代变量 q 遍历 1:Q
    B(q,:) = mixmat(q,:) * permute(B2(q,:,:), [2 3 1]);  % 详解: 调用函数：B(q,:) = mixmat(q,:) * permute(B2(q,:,:), [2 3 1])
  end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
    B(:,t) = sum(mixmat .* B2(:,:,t), 2);  % 详解: 调用函数：B(:,t) = sum(mixmat .* B2(:,:,t), 2)
  end  % 详解: 执行语句
end  % 详解: 执行语句





