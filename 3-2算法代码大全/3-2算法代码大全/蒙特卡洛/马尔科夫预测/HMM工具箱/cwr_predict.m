% 文件: cwr_predict.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function  [mu, Sigma, weights, mask] = cwr_predict(cwr, X, mask_data)  % 详解: 函数定义：cwr_predict(cwr, X, mask_data), 返回：mu, Sigma, weights, mask

[nx T] = size(X);  % 详解: 获取向量/矩阵尺寸
[ny nx nc] = size(cwr.weightsY);  % 详解: 获取向量/矩阵尺寸
mu = zeros(ny, T);  % 详解: 赋值：将 zeros(...) 的结果保存到 mu
Sigma = zeros(ny, ny, T);  % 详解: 赋值：将 zeros(...) 的结果保存到 Sigma

if nargout == 4  % 详解: 条件判断：if (nargout == 4)
  comp_mask = 1;  % 详解: 赋值：计算表达式并保存到 comp_mask
  N = size(mask_data,2);  % 详解: 赋值：将 size(...) 的结果保存到 N
  mask = zeros(N,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 mask
else  % 详解: 条件判断：else 分支
  comp_mask = 0;  % 详解: 赋值：计算表达式并保存到 comp_mask
end  % 详解: 执行语句

if nc==1  % 详解: 条件判断：if (nc==1)
  if isempty(cwr.weightsY)  % 详解: 条件判断：if (isempty(cwr.weightsY))
    mu = repmat(cwr.muY, 1, T);  % 详解: 赋值：将 repmat(...) 的结果保存到 mu
    Sigma = repmat(cwr.SigmaY, [1 1 T]);  % 详解: 赋值：将 repmat(...) 的结果保存到 Sigma
  else  % 详解: 条件判断：else 分支
    mu = repmat(cwr.muY, 1, T) + cwr.weightsY * X;  % 详解: 赋值：将 repmat(...) 的结果保存到 mu
    Sigma = repmat(cwr.SigmaY, [1 1 T]);  % 详解: 赋值：将 repmat(...) 的结果保存到 Sigma
  end  % 详解: 执行语句
  if comp_mask, mask = gaussian_prob(mask_data, mu, Sigma); end  % 详解: 条件判断：if (comp_mask, mask = gaussian_prob(mask_data, mu, Sigma); end)
  weights = [];  % 详解: 赋值：计算表达式并保存到 weights
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句


likX = mixgauss_prob(X, cwr.muX, cwr.SigmaX);  % 详解: 赋值：将 mixgauss_prob(...) 的结果保存到 likX
weights = normalize(repmat(cwr.priorC, 1, T) .* likX, 1);  % 详解: 赋值：将 normalize(...) 的结果保存到 weights
for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
  mut = zeros(ny, nc);  % 详解: 赋值：将 zeros(...) 的结果保存到 mut
  for c=1:nc  % 详解: for 循环：迭代变量 c 遍历 1:nc
    mut(:,c) = cwr.muY(:,c) + cwr.weightsY(:,:,c)*X(:,t);  % 详解: 调用函数：mut(:,c) = cwr.muY(:,c) + cwr.weightsY(:,:,c)*X(:,t)
    if comp_mask  % 详解: 条件判断：if (comp_mask)
      mask = mask + gaussian_prob(mask_data, mut(:,c), cwr.SigmaY(:,:,c)) * weights(c);  % 详解: 赋值：计算表达式并保存到 mask
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  [mu(:,t), Sigma(:,:,t)] = collapse_mog(mut, cwr.SigmaY, weights(:,t));  % 详解: 执行语句
end  % 详解: 执行语句




