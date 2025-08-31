% 文件: mixgauss_prob_test.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function test_eval_pdf_cond_mixgauss()  % 详解: 函数定义：test_eval_pdf_cond_mixgauss()

Q = 2; M = 3; d = 4; T = 5;  % 详解: 赋值：计算表达式并保存到 Q

mu = rand(d,Q,M);  % 详解: 赋值：将 rand(...) 的结果保存到 mu
data = randn(d,T);  % 详解: 赋值：将 randn(...) 的结果保存到 data
mixmat = mk_stochastic(ones(Q,M));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 mixmat

Sigma = 0.01;  % 详解: 赋值：计算表达式并保存到 Sigma

mu = rand(d,M,Q);  % 详解: 赋值：将 rand(...) 的结果保存到 mu
weights = mixmat';  % 赋值：设置变量 weights  % 详解: 赋值：计算表达式并保存到 weights  % 详解: 赋值：计算表达式并保存到 weights
N = M*ones(1,Q);  % 详解: 赋值：计算表达式并保存到 N
tic; [B, B2, D] = parzen(data, mu, Sigma, N, weights); toc  % 详解: 执行语句
tic; [BC, B2C, DC] = parzenC(data, mu, Sigma, N); toc  % 详解: 执行语句
approxeq(B,BC)  % 详解: 调用函数：approxeq(B,BC)
B2C = reshape(B2C,[M Q T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 B2C
approxeq(B2,B2C)  % 详解: 调用函数：approxeq(B2,B2C)
DC = reshape(DC,[M Q T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 DC
approxeq(D,DC)  % 详解: 调用函数：approxeq(D,DC)


return  % 详解: 返回：从当前函数返回

tic; [B, B2] = eval_pdf_cond_mixgauss(data, mu, Sigma, mixmat); toc  % 详解: 执行语句
tic; C = eval_pdf_cond_parzen(data, mu, Sigma); toc  % 详解: 执行语句
approxeq(B,C)  % 详解: 调用函数：approxeq(B,C)

return;  % 详解: 返回：从当前函数返回


mu = reshape(mu, [d Q*M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu

data = mk_unit_norm(data);  % 详解: 赋值：将 mk_unit_norm(...) 的结果保存到 data
mu = mk_unit_norm(mu);  % 详解: 赋值：将 mk_unit_norm(...) 的结果保存到 mu
tic; D = 2 -2*(data'*mu); toc % avoid an expensive repmat  % 中文: tic; d = 2 -2*（data'*mu）'; toc  % 详解: 执行语句  % 详解: 执行语句
tic; D2 = sqdist(data, mu); toc  % 详解: 执行语句
approxeq(D,D2)  % 详解: 调用函数：approxeq(D,D2)


mu = reshape(mu, [d Q*M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu
D = dist2(data', mu');  % 详解: 赋值：将 dist2(...) 的结果保存到 D
denom = (2*pi*Sigma)^(d/2);  % 详解: 赋值：计算表达式并保存到 denom
numer = exp(-0.5/Sigma  * D');  % 赋值：设置变量 numer  % 详解: 赋值：将 exp(...) 的结果保存到 numer  % 详解: 赋值：将 exp(...) 的结果保存到 numer
B2 = numer / denom;  % 详解: 赋值：计算表达式并保存到 B2
B2 = reshape(B2, [Q M T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 B2

tic; B = squeeze(sum(B2 .* repmat(mixmat, [1 1 T]), 2)); toc  % 详解: 统计：求和/均值/中位数

tic  % 详解: 执行语句
A = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 A
for q=1:Q  % 详解: for 循环：迭代变量 q 遍历 1:Q
  A(q,:) = mixmat(q,:) * squeeze(B2(q,:,:));  % 详解: 调用函数：A(q,:) = mixmat(q,:) * squeeze(B2(q,:,:))
end  % 详解: 执行语句
toc  % 详解: 执行语句
assert(approxeq(A,B))  % 详解: 调用函数：assert(approxeq(A,B))

tic  % 详解: 执行语句
A = zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 A
for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
  A(:,t) = sum(mixmat .* B2(:,:,t), 2);  % 详解: 调用函数：A(:,t) = sum(mixmat .* B2(:,:,t), 2)
end  % 详解: 执行语句
toc  % 详解: 执行语句
assert(approxeq(A,B))  % 详解: 调用函数：assert(approxeq(A,B))

    


mu = reshape(mu, [d Q M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu
B3 = zeros(Q,M,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 B3
for j=1:Q  % 详解: for 循环：迭代变量 j 遍历 1:Q
  for k=1:M  % 详解: for 循环：迭代变量 k 遍历 1:M
    B3(j,k,:) = gaussian_prob(data, mu(:,j,k), Sigma*eye(d));  % 详解: 调用函数：B3(j,k,:) = gaussian_prob(data, mu(:,j,k), Sigma*eye(d))
  end  % 详解: 执行语句
end  % 详解: 执行语句
assert(approxeq(B2, B3))  % 详解: 调用函数：assert(approxeq(B2, B3))

logB4 = -(d/2)*log(2*pi*Sigma) - (1/(2*Sigma))*D;  % 详解: 赋值：计算表达式并保存到 logB4
B4 = reshape(exp(logB4), [Q M T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 B4
assert(approxeq(B4, B3))  % 详解: 调用函数：assert(approxeq(B4, B3))



  

Sigma = rand_psd(d,d);  % 详解: 赋值：将 rand_psd(...) 的结果保存到 Sigma
mu = reshape(mu, [d Q*M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu
D = sqdist(data, mu, inv(Sigma))';  % 矩阵求逆  % 详解: 赋值：将 sqdist(...) 的结果保存到 D  % 详解: 赋值：将 sqdist(...) 的结果保存到 D
denom = sqrt(det(2*pi*Sigma));  % 详解: 赋值：将 sqrt(...) 的结果保存到 denom
numer = exp(-0.5 * D);  % 详解: 赋值：将 exp(...) 的结果保存到 numer
B2 = numer / denom;  % 详解: 赋值：计算表达式并保存到 B2
B2 = reshape(B2, [Q M T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 B2

mu = reshape(mu, [d Q M]);  % 详解: 赋值：将 reshape(...) 的结果保存到 mu
B3 = zeros(Q,M,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 B3
for j=1:Q  % 详解: for 循环：迭代变量 j 遍历 1:Q
  for k=1:M  % 详解: for 循环：迭代变量 k 遍历 1:M
    B3(j,k,:) = gaussian_prob(data, mu(:,j,k), Sigma);  % 详解: 调用函数：B3(j,k,:) = gaussian_prob(data, mu(:,j,k), Sigma)
  end  % 详解: 执行语句
end  % 详解: 执行语句
assert(approxeq(B2, B3))  % 详解: 调用函数：assert(approxeq(B2, B3))

logB4 = -(d/2)*log(2*pi) - 0.5*logdet(Sigma) - 0.5*D;  % 详解: 赋值：计算表达式并保存到 logB4
B4 = reshape(exp(logB4), [Q M T]);  % 详解: 赋值：将 reshape(...) 的结果保存到 B4
assert(approxeq(B4, B3))  % 详解: 调用函数：assert(approxeq(B4, B3))




