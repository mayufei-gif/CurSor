% 文件: linear_regression.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [muY, SigmaY, weightsY]  = linear_regression(X, Y, varargin)  % 详解: 函数定义：linear_regression(X, Y, varargin), 返回：muY, SigmaY, weightsY

[cov_typeY, clamp_weights,  muY, SigmaY, weightsY,...  % 详解: 执行语句
 cov_priorY,  regress, clamp_covY] = process_options(...  % 详解: 执行语句
    varargin, ...  % 详解: 执行语句
     'cov_typeY', 'full', 'clamp_weights', 0, ...  % 详解: 执行语句
     'muY', [], 'SigmaY', [], 'weightsY', [], ...  % 详解: 执行语句
     'cov_priorY', [], 'regress', 1,  'clamp_covY', 0);  % 详解: 执行语句
     
[nx N] = size(X);  % 详解: 获取向量/矩阵尺寸
[ny N2] = size(Y);  % 详解: 获取向量/矩阵尺寸
if N ~= N2  % 详解: 条件判断：if (N ~= N2)
  error(sprintf('nsamples X (%d) ~= nsamples Y (%d)', N, N2));  % 详解: 调用函数：error(sprintf('nsamples X (%d) ~= nsamples Y (%d)', N, N2))
end  % 详解: 执行语句

w = 1/N;  % 详解: 赋值：计算表达式并保存到 w
WYbig = Y*w;  % 详解: 赋值：计算表达式并保存到 WYbig
WYY = WYbig * Y';   % 赋值：设置变量 WYY  % 详解: 赋值：计算表达式并保存到 WYY  % 详解: 赋值：计算表达式并保存到 WYY
WY = sum(WYbig, 2);  % 详解: 赋值：将 sum(...) 的结果保存到 WY
WYTY = sum(diag(WYbig' * Y));  % 统计：求和/均值/中位数  % 详解: 赋值：将 sum(...) 的结果保存到 WYTY  % 详解: 赋值：将 sum(...) 的结果保存到 WYTY
if ~regress  % 详解: 条件判断：if (~regress)
  weightsY = [];  % 详解: 赋值：计算表达式并保存到 weightsY
  [muY, SigmaY] = ...  % 详解: 执行语句
      mixgauss_Mstep(1, WY, WYY, WYTY, ...  % 详解: 执行语句
		     'cov_type', cov_typeY, 'cov_prior', cov_priorY);  % 详解: 执行语句
  assert(approxeq(muY, mean(Y')))  % 统计：求和/均值/中位数  % 详解: 调用函数：assert(approxeq(muY, mean(Y')))  % 详解: 调用函数：assert(approxeq(muY, mean(Y'))) % 统计：求和/均值/中位数 % 详解: 调用函数：assert(approxeq(muY, mean(Y')))
  assert(approxeq(SigmaY, cov(Y') + 0.01*eye(ny)))  % 创建单位矩阵  % 详解: 调用函数：assert(approxeq(SigmaY, cov(Y') + 0.01*eye(ny)))  % 详解: 调用函数：assert(approxeq(SigmaY, cov(Y') + 0.01*eye(ny))) % 创建单位矩阵 % 详解: 调用函数：assert(approxeq(SigmaY, cov(Y') + 0.01*eye(ny)))
else  % 详解: 条件判断：else 分支
  WXbig = X*w;  % 详解: 赋值：计算表达式并保存到 WXbig
  WXX = WXbig * X';  % 赋值：设置变量 WXX  % 详解: 赋值：计算表达式并保存到 WXX  % 详解: 赋值：计算表达式并保存到 WXX
  WX = sum(WXbig, 2);  % 详解: 赋值：将 sum(...) 的结果保存到 WX
  WXTX = sum(diag(WXbig' * X));  % 统计：求和/均值/中位数  % 详解: 赋值：将 sum(...) 的结果保存到 WXTX  % 详解: 赋值：将 sum(...) 的结果保存到 WXTX
  WXY = WXbig * Y';  % 赋值：设置变量 WXY  % 详解: 赋值：计算表达式并保存到 WXY  % 详解: 赋值：计算表达式并保存到 WXY
  [muY, SigmaY, weightsY] = ...  % 详解: 执行语句
      clg_Mstep(1, WY, WYY, WYTY, WX, WXX, WXY, ...  % 详解: 执行语句
		'cov_type', cov_typeY, 'cov_prior', cov_priorY);  % 详解: 执行语句
end  % 详解: 执行语句
if clamp_covY, SigmaY = SigmaY; end  % 详解: 条件判断：if (clamp_covY, SigmaY = SigmaY; end)
if clamp_weights,  weightsY = weightsY; end  % 详解: 条件判断：if (clamp_weights,  weightsY = weightsY; end)

if nx==1 & ny==1 & regress  % 详解: 条件判断：if (nx==1 & ny==1 & regress)
  P = polyfit(X,Y);  % 详解: 赋值：将 polyfit(...) 的结果保存到 P
  assert(approxeq(muY, P(2)))  % 详解: 调用函数：assert(approxeq(muY, P(2)))
  assert(approxeq(weightsY, P(1)))  % 详解: 调用函数：assert(approxeq(weightsY, P(1)))
end  % 详解: 执行语句

if 0  % 详解: 条件判断：if (0)
  c1 = randn(2,100);   c2 = randn(2,100);  % 详解: 赋值：将 randn(...) 的结果保存到 c1
  y = c2(1,:); X = [ones(size(c1,2),1) c1'];  % 创建全 1 矩阵/数组  % 详解: 赋值：将 c2(...) 的结果保存到 y  % 详解: 赋值：将 c2(...) 的结果保存到 y
  b = regress(y(:), X);  % 详解: 赋值：将 regress(...) 的结果保存到 b
  [m,s,w] = linear_regression(c1, y);  % 详解: 执行语句
  assert(approxeq(b(1),m))  % 详解: 调用函数：assert(approxeq(b(1),m))
  assert(approxeq(b(2), w(1)))  % 详解: 调用函数：assert(approxeq(b(2), w(1)))
  assert(approxeq(b(3), w(2)))  % 详解: 调用函数：assert(approxeq(b(3), w(2)))
end  % 详解: 执行语句




