% 文件: parzen.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [B,B2,dist] = parzen(data, mu, Sigma, N)  % 详解: 函数定义：parzen(data, mu, Sigma, N), 返回：B,B2,dist


if nargout >= 2  % 详解: 条件判断：if (nargout >= 2)
  keep_B2 = 1;  % 详解: 赋值：计算表达式并保存到 keep_B2
else  % 详解: 条件判断：else 分支
  keep_B2 = 0;  % 详解: 赋值：计算表达式并保存到 keep_B2
end  % 详解: 执行语句

if nargout >= 3  % 详解: 条件判断：if (nargout >= 3)
  keep_dist = 1;  % 详解: 赋值：计算表达式并保存到 keep_dist
else  % 详解: 条件判断：else 分支
  keep_dist = 0;  % 详解: 赋值：计算表达式并保存到 keep_dist
end  % 详解: 执行语句

[d M Q] = size(mu);  % 详解: 获取向量/矩阵尺寸
[d T] = size(data);  % 详解: 获取向量/矩阵尺寸

M = max(N(:));  % 详解: 赋值：将 max(...) 的结果保存到 M

B = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 B
const1 = (2*pi*Sigma)^(-d/2);  % 详解: 赋值：计算表达式并保存到 const1
const2 = -(1/(2*Sigma));  % 详解: 赋值：计算表达式并保存到 const2
if T*Q*M>20000000  % 详解: 条件判断：if (T*Q*M>20000000)
  disp('eval parzen for loop')  % 详解: 调用函数：disp('eval parzen for loop')
  if keep_dist,  % 详解: 条件判断：if (keep_dist,)
    dist = zeros(M,Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 dist
  end  % 详解: 执行语句
  if keep_B2  % 详解: 条件判断：if (keep_B2)
    B2 = zeros(M,Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 B2
  end  % 详解: 执行语句
  for q=1:Q  % 详解: for 循环：迭代变量 q 遍历 1:Q
    D = sqdist(mu(:,1:N(q),q), data);  % 详解: 赋值：将 sqdist(...) 的结果保存到 D
    if keep_dist  % 详解: 条件判断：if (keep_dist)
      dist(:,q,:) = D;  % 详解: 执行语句
    end  % 详解: 执行语句
    tmp = const1 * exp(const2*D);  % 详解: 赋值：计算表达式并保存到 tmp
    if keep_B2,  % 详解: 条件判断：if (keep_B2,)
      B2(:,q,:) = tmp;  % 详解: 执行语句
    end  % 详解: 执行语句
    if N(q) > 0  % 详解: 条件判断：if (N(q) > 0)
      B(q,:) = (1/N(q)) * sum(tmp,1);  % 详解: 调用函数：B(q,:) = (1/N(q)) * sum(tmp,1)
    end  % 详解: 执行语句
  end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  dist = sqdist(reshape(mu(:,1:M,:), [d M*Q]), data);  % 详解: 赋值：将 sqdist(...) 的结果保存到 dist
  dist = reshape(dist, [M Q T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 dist
  B2 = const1 * exp(const2*dist);  % 详解: 赋值：计算表达式并保存到 B2
  if ~keep_dist  % 详解: 条件判断：if (~keep_dist)
    clear dist  % 详解: 执行语句
  end  % 详解: 执行语句
  
   
  Ns = repmat(N(:)', [M 1]);  % 赋值：设置变量 Ns  % 详解: 赋值：将 repmat(...) 的结果保存到 Ns  % 详解: 赋值：将 repmat(...) 的结果保存到 Ns
  ramp = 1:M;  % 详解: 赋值：计算表达式并保存到 ramp
  ramp = repmat(ramp(:), [1 Q]);  % 详解: 赋值：将 repmat(...) 的结果保存到 ramp
  n = N + (N==0);  % 详解: 赋值：计算表达式并保存到 n
  N1 = repmat(1 ./ n(:)', [M 1]);  % 赋值：设置变量 N1  % 详解: 赋值：将 repmat(...) 的结果保存到 N1  % 详解: 赋值：将 repmat(...) 的结果保存到 N1
  mask = (ramp <= Ns);  % 详解: 赋值：计算表达式并保存到 mask
  weights = N1 .* mask;  % 详解: 赋值：计算表达式并保存到 weights
  B2 = B2 .* repmat(mask, [1 1 T]);  % 详解: 赋值：计算表达式并保存到 B2
  
  B = squeeze(sum(B2 .* repmat(weights, [1 1 T]), 1));  % 详解: 赋值：将 squeeze(...) 的结果保存到 B
  B = reshape(B, [Q T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 B
end  % 详解: 执行语句

  





