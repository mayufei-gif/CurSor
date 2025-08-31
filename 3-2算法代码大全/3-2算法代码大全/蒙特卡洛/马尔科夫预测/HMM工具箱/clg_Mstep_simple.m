% 文件: clg_Mstep_simple.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [mu, B] = clg_Mstep_simple(w, Y, YY, YTY, X, XX, XY)  % 详解: 函数定义：clg_Mstep_simple(w, Y, YY, YTY, X, XX, XY), 返回：mu, B

[Ysz Q] = size(Y);  % 详解: 获取向量/矩阵尺寸

if isempty(X)  % 详解: 条件判断：if (isempty(X))
  B2 = zeros(Ysz, 1, Q);  % 详解: 赋值：将 zeros(...) 的结果保存到 B2
  for i=1:Q  % 详解: for 循环：迭代变量 i 遍历 1:Q
    B(:,:,i) = B2(:,1:0,i);  % 详解: 调用函数：B(:,:,i) = B2(:,1:0,i)
  end  % 详解: 执行语句
  [mu, Sigma] = mixgauss_Mstep(w, Y, YY, YTY);  % 详解: 执行语句
  return;  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

N = sum(w);  % 详解: 赋值：将 sum(...) 的结果保存到 N

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




