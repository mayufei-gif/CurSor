% 文件: mixgauss_Mstep.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [mu, Sigma] = mixgauss_Mstep(w, Y, YY, YTY, varargin)  % 详解: 函数定义：mixgauss_Mstep(w, Y, YY, YTY, varargin), 返回：mu, Sigma

[cov_type, tied_cov,  clamped_cov, clamped_mean, cov_prior, other] = ...  % 详解: 执行语句
    process_options(varargin,...  % 详解: 执行语句
		    'cov_type', 'full', 'tied_cov', 0,  'clamped_cov', [], 'clamped_mean', [], ...  % 详解: 执行语句
		    'cov_prior', []);  % 详解: 执行语句

[Ysz Q] = size(Y);  % 详解: 获取向量/矩阵尺寸
N = sum(w);  % 详解: 赋值：将 sum(...) 的结果保存到 N
if isempty(cov_prior)  % 详解: 条件判断：if (isempty(cov_prior))
  cov_prior = repmat(0.01*eye(Ysz,Ysz), [1 1 Q]);  % 详解: 赋值：将 repmat(...) 的结果保存到 cov_prior
end  % 详解: 执行语句
YY = reshape(YY, [Ysz Ysz Q]);  % 详解: 赋值：将 reshape(...) 的结果保存到 YY

w = w + (w==0);  % 详解: 赋值：计算表达式并保存到 w
		    
if ~isempty(clamped_mean)  % 详解: 条件判断：if (~isempty(clamped_mean))
  mu = clamped_mean;  % 详解: 赋值：计算表达式并保存到 mu
else  % 详解: 条件判断：else 分支
  mu = zeros(Ysz, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 mu
  for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
    mu(:,i) = Y(:,i) / w(i);  % 详解: 调用函数：mu(:,i) = Y(:,i) / w(i)
  end  % 详解: 执行语句
end  % 详解: 执行语句

if ~isempty(clamped_cov)  % 详解: 条件判断：if (~isempty(clamped_cov))
  Sigma = clamped_cov;  % 详解: 赋值：计算表达式并保存到 Sigma
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

if ~tied_cov  % 详解: 条件判断：if (~tied_cov)
  Sigma = zeros(Ysz,Ysz,Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 Sigma
  for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
    if cov_type(1) == 's'  % 详解: 条件判断：if (cov_type(1) == 's')
      s2 = (1/Ysz)*( (YTY(i)/w(i)) - mu(:,i)'*mu(:,i) );  % 赋值：设置变量 s2  % 详解: 赋值：计算表达式并保存到 s2  % 详解: 赋值：计算表达式并保存到 s2
      Sigma(:,:,i) = s2 * eye(Ysz);  % 详解: 调用函数：Sigma(:,:,i) = s2 * eye(Ysz)
    else  % 详解: 条件判断：else 分支
      SS = YY(:,:,i)/w(i)  - mu(:,i)*mu(:,i)';  % 赋值：设置变量 SS  % 详解: 赋值：将 YY(...) 的结果保存到 SS  % 详解: 赋值：将 YY(...) 的结果保存到 SS
      if cov_type(1)=='d'  % 详解: 条件判断：if (cov_type(1)=='d')
	SS = diag(diag(SS));  % 详解: 赋值：将 diag(...) 的结果保存到 SS
      end  % 详解: 执行语句
      Sigma(:,:,i) = SS;  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  if cov_type(1) == 's'  % 详解: 条件判断：if (cov_type(1) == 's')
    s2 = (1/(N*Ysz))*(sum(YTY,2) + sum(diag(mu'*mu) .* w));  % 统计：求和/均值/中位数  % 详解: 赋值：计算表达式并保存到 s2  % 详解: 赋值：计算表达式并保存到 s2
    Sigma = s2*eye(Ysz);  % 详解: 赋值：计算表达式并保存到 Sigma
  else  % 详解: 条件判断：else 分支
    SS = zeros(Ysz, Ysz);  % 详解: 赋值：将 zeros(...) 的结果保存到 SS
    for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
      SS = SS + YY(:,:,i)/N - mu(:,i)*mu(:,i)';  % 赋值：设置变量 SS  % 详解: 赋值：计算表达式并保存到 SS  % 详解: 赋值：计算表达式并保存到 SS
    end  % 详解: 执行语句
    if cov_type(1) == 'd'  % 详解: 条件判断：if (cov_type(1) == 'd')
      Sigma = diag(diag(SS));  % 详解: 赋值：将 diag(...) 的结果保存到 Sigma
    else  % 详解: 条件判断：else 分支
      Sigma = SS;  % 详解: 赋值：计算表达式并保存到 Sigma
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

if tied_cov  % 详解: 条件判断：if (tied_cov)
  Sigma =  repmat(Sigma, [1 1 Q]);  % 详解: 赋值：将 repmat(...) 的结果保存到 Sigma
end  % 详解: 执行语句
Sigma = Sigma + cov_prior;  % 详解: 赋值：计算表达式并保存到 Sigma




