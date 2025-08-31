% 文件: clg_Mstep.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [mu, Sigma, B] = clg_Mstep(w, Y, YY, YTY, X, XX, XY, varargin)  % 详解: 函数定义：clg_Mstep(w, Y, YY, YTY, X, XX, XY, varargin), 返回：mu, Sigma, B

[cov_type, tied_cov, ...  % 详解: 执行语句
 clamped_cov, clamped_mean, clamped_weights,  cov_prior, ...  % 详解: 执行语句
 xs, ys, post] = ...  % 详解: 执行语句
    process_options(varargin, ...  % 详解: 执行语句
		    'cov_type', 'full', 'tied_cov', 0,  'clamped_cov', [], 'clamped_mean', [], ...  % 详解: 执行语句
		    'clamped_weights', [], 'cov_prior', [], ...  % 详解: 执行语句
		    'xs', [], 'ys', [], 'post', []);  % 详解: 执行语句

[Ysz Q] = size(Y);  % 详解: 获取向量/矩阵尺寸

if isempty(X)  % 详解: 条件判断：if (isempty(X))
  B2 = zeros(Ysz, 1, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 B2
  for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
    B(:,:,i) = B2(:,1:0,i);  % 详解: 调用函数：B(:,:,i) = B2(:,1:0,i)
  end  % 详解: 执行语句
  [mu, Sigma] = mixgauss_Mstep(w, Y, YY, YTY, varargin{:});  % 详解: 执行语句
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句


N = sum(w);  % 详解: 赋值：将 sum(...) 的结果保存到 N
if isempty(cov_prior)  % 详解: 条件判断：if (isempty(cov_prior))
  cov_prior = 0.01*repmat(eye(Ysz,Ysz), [1 1 Q]);  % 详解: 赋值：计算表达式并保存到 cov_prior
end  % 详解: 执行语句

w = w + (w==0);  % 详解: 赋值：计算表达式并保存到 w

Xsz = size(X,1);  % 详解: 赋值：将 size(...) 的结果保存到 Xsz
ZZ = zeros(Xsz+1, Xsz+1, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 ZZ
ZY = zeros(Xsz+1, Ysz, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 ZY
for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
  ZZ(:,:,i) = [XX(:,:,i)  X(:,i);  % 详解: 调用函数：ZZ(:,:,i) = [XX(:,:,i) X(:,i)
	       X(:,i)'    w(i)];  % 调用函数：X  % 详解: 执行语句  % 详解: 执行语句
  ZY(:,:,i) = [XY(:,:,i);  % 详解: 调用函数：ZY(:,:,i) = [XY(:,:,i)
	       Y(:,i)'];  % 调用函数：Y  % 详解: 执行语句  % 详解: 执行语句
end  % 详解: 执行语句



if ~isempty(clamped_weights) & ~isempty(clamped_mean)  % 详解: 条件判断：if (~isempty(clamped_weights) & ~isempty(clamped_mean))
  B = clamped_weights;  % 详解: 赋值：计算表达式并保存到 B
  mu = clamped_mean;  % 详解: 赋值：计算表达式并保存到 mu
end  % 详解: 执行语句
if ~isempty(clamped_weights) & isempty(clamped_mean)  % 详解: 条件判断：if (~isempty(clamped_weights) & isempty(clamped_mean))
  B = clamped_weights;  % 详解: 赋值：计算表达式并保存到 B
  mu = zeros(Ysz, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 mu
  for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
    mu(:,i) = (Y(:,i) - B(:,:,i)*X(:,i)) / w(i);  % 详解: 调用函数：mu(:,i) = (Y(:,i) - B(:,:,i)*X(:,i)) / w(i)
  end  % 详解: 执行语句
end  % 详解: 执行语句
if isempty(clamped_weights) & ~isempty(clamped_mean)  % 详解: 条件判断：if (isempty(clamped_weights) & ~isempty(clamped_mean))
  mu = clamped_mean;  % 详解: 赋值：计算表达式并保存到 mu
  B = zeros(Ysz, Xsz, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 B
  for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
    tmp = XY(:,:,i)' - mu(:,i)*X(:,i)';  % 详解: 赋值：将 XY(...) 的结果保存到 tmp
    B(:,:,i) = (XX(:,:,i) \ tmp')';  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句
if isempty(clamped_weights) & isempty(clamped_mean)  % 详解: 条件判断：if (isempty(clamped_weights) & isempty(clamped_mean))
  mu = zeros(Ysz, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 mu
  B = zeros(Ysz, Xsz, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 B
  for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
    if rcond(ZZ(:,:,i)) < 1e-10  % 详解: 条件判断：if (rcond(ZZ(:,:,i)) < 1e-10)
      sprintf('clg_Mstep warning: ZZ(:,:,%d) is ill-conditioned', i);  % 详解: 调用函数：sprintf('clg_Mstep warning: ZZ(:,:,%d) is ill-conditioned', i)
      ZZ(:,:,i) = ZZ(:,:,i) + 1e-5*eye(Xsz+1);  % 详解: 调用函数：ZZ(:,:,i) = ZZ(:,:,i) + 1e-5*eye(Xsz+1)
    end  % 详解: 执行语句
    A = (ZZ(:,:,i) \ ZY(:,:,i))';  % 赋值：设置变量 A  % 详解: 赋值：计算表达式并保存到 A  % 详解: 赋值：计算表达式并保存到 A
    B(:,:,i) = A(:, 1:Xsz);  % 详解: 调用函数：B(:,:,i) = A(:, 1:Xsz)
    mu(:,i) = A(:, Xsz+1);  % 详解: 调用函数：mu(:,i) = A(:, Xsz+1)
  end  % 详解: 执行语句
end  % 详解: 执行语句

if ~isempty(clamped_cov)  % 详解: 条件判断：if (~isempty(clamped_cov))
  Sigma = clamped_cov;  % 详解: 赋值：计算表达式并保存到 Sigma
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句



if cov_type(1)=='s'  % 详解: 条件判断：if (cov_type(1)=='s')
  if ~tied_cov  % 详解: 条件判断：if (~tied_cov)
    Sigma = zeros(Ysz, Ysz, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 Sigma
    for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
      A = [B(:,:,i) mu(:,i)];  % 详解: 赋值：计算表达式并保存到 A
      s = (YTY(i) + trace(A'*A*ZZ(:,:,i)) - trace(2*A*ZY(:,:,i))) / (Ysz*w(i));  % 赋值：设置变量 s  % 详解: 赋值：计算表达式并保存到 s  % 详解: 赋值：计算表达式并保存到 s
      Sigma(:,:,i) = s*eye(Ysz,Ysz);  % 详解: 调用函数：Sigma(:,:,i) = s*eye(Ysz,Ysz)

      if ~isempty(xs)  % 详解: 条件判断：if (~isempty(xs))
	[nx T] = size(xs);  % 详解: 获取向量/矩阵尺寸
	zs = [xs; ones(1,T)];  % 详解: 赋值：计算表达式并保存到 zs
	yty = 0;  % 详解: 赋值：计算表达式并保存到 yty
	zAAz = 0;  % 详解: 赋值：计算表达式并保存到 zAAz
	yAz = 0;  % 详解: 赋值：计算表达式并保存到 yAz
	for t=1:T  % 详解: for 循环：迭代变量 t 遍历 1:T
	  yty = yty + ys(:,t)'*ys(:,t) * post(i,t);  % 赋值：设置变量 yty  % 详解: 赋值：计算表达式并保存到 yty  % 详解: 赋值：计算表达式并保存到 yty
	  zAAz = zAAz + zs(:,t)'*A'*A*zs(:,t)*post(i,t);  % 详解: 赋值：计算表达式并保存到 zAAz
	  yAz = yAz + ys(:,t)'*A*zs(:,t)*post(i,t);  % 赋值：设置变量 yAz  % 详解: 赋值：计算表达式并保存到 yAz  % 详解: 赋值：计算表达式并保存到 yAz
	end  % 详解: 执行语句
	assert(approxeq(yty, YTY(i)))  % 详解: 调用函数：assert(approxeq(yty, YTY(i)))
	assert(approxeq(zAAz, trace(A'*A*ZZ(:,:,i))))  % 调用函数：assert  % 详解: 调用函数：assert(approxeq(zAAz, trace(A'*A*ZZ(:,:,i))))  % 详解: 调用函数：assert(approxeq(zAAz, trace(A'*A*ZZ(:,:,i)))) % 调用函数：assert % 详解: 调用函数：assert(approxeq(zAAz, trace(A'*A*ZZ(:,:,i))))
	assert(approxeq(yAz, trace(A*ZY(:,:,i))))  % 详解: 调用函数：assert(approxeq(yAz, trace(A*ZY(:,:,i))))
	s2 = (yty + zAAz - 2*yAz) / (Ysz*w(i));  % 详解: 赋值：计算表达式并保存到 s2
	assert(approxeq(s,s2))  % 详解: 调用函数：assert(approxeq(s,s2))
      end  % 详解: 执行语句
      
    end  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    S = 0;  % 详解: 赋值：计算表达式并保存到 S
    for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
      A = [B(:,:,i) mu(:,i)];  % 详解: 赋值：计算表达式并保存到 A
      S = S + trace(YTY(i) + A'*A*ZZ(:,:,i) - 2*A*ZY(:,:,i));  % 赋值：设置变量 S  % 详解: 赋值：计算表达式并保存到 S  % 详解: 赋值：计算表达式并保存到 S
    end  % 详解: 执行语句
    Sigma = repmat(S / (N*Ysz), [1 1 Q]);  % 详解: 赋值：将 repmat(...) 的结果保存到 Sigma
  end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  if ~tied_cov  % 详解: 条件判断：if (~tied_cov)
    Sigma = zeros(Ysz, Ysz, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 Sigma
    for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
      A = [B(:,:,i) mu(:,i)];  % 详解: 赋值：计算表达式并保存到 A
      SS = (YY(:,:,i) - ZY(:,:,i)'*A' - A*ZY(:,:,i) + A*ZZ(:,:,i)*A') / w(i);  % 赋值：设置变量 SS  % 详解: 赋值：计算表达式并保存到 SS  % 详解: 赋值：计算表达式并保存到 SS
      if cov_type(1)=='d'  % 详解: 条件判断：if (cov_type(1)=='d')
	Sigma(:,:,i) = diag(diag(SS));  % 详解: 调用函数：Sigma(:,:,i) = diag(diag(SS))
      else  % 详解: 条件判断：else 分支
	Sigma(:,:,i) = SS;  % 详解: 执行语句
      end  % 详解: 执行语句
    end  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    SS = zeros(Ysz, Ysz);  % 详解: 赋值：将 zeros(...) 的结果保存到 SS
    for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
      A = [B(:,:,i) mu(:,i)];  % 详解: 赋值：计算表达式并保存到 A
      SS = SS + (YY(:,:,i) - ZY(:,:,i)'*A' - A*ZY(:,:,i) + A*ZZ(:,:,i)*A');  % 赋值：设置变量 SS  % 详解: 赋值：计算表达式并保存到 SS  % 详解: 赋值：计算表达式并保存到 SS
    end  % 详解: 执行语句
    SS = SS / N;  % 详解: 赋值：计算表达式并保存到 SS
    if cov_type(1)=='d'  % 详解: 条件判断：if (cov_type(1)=='d')
      Sigma = diag(diag(SS));  % 详解: 赋值：将 diag(...) 的结果保存到 Sigma
    else  % 详解: 条件判断：else 分支
      Sigma = SS;  % 详解: 赋值：计算表达式并保存到 Sigma
    end  % 详解: 执行语句
    Sigma = repmat(Sigma, [1 1 Q]);  % 详解: 赋值：将 repmat(...) 的结果保存到 Sigma
  end  % 详解: 执行语句
end  % 详解: 执行语句

Sigma = Sigma + cov_prior;  % 详解: 赋值：计算表达式并保存到 Sigma
  




