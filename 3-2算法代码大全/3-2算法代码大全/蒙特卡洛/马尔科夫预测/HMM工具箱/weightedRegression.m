% 文件: weightedRegression.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [a, b, error] = weightedRegression(x, z, w)  % 详解: 函数定义：weightedRegression(x, z, w), 返回：a, b, error

if nargin < 3, w = ones(1,length(x)); end  % 详解: 条件判断：if (nargin < 3, w = ones(1,length(x)); end)

w = w(:)';  % 赋值：设置变量 w  % 详解: 赋值：将 w(...) 的结果保存到 w  % 详解: 赋值：将 w(...) 的结果保存到 w
x = x(:)';  % 赋值：设置变量 x  % 详解: 赋值：将 x(...) 的结果保存到 x  % 详解: 赋值：将 x(...) 的结果保存到 x
z = z(:)';  % 赋值：设置变量 z  % 详解: 赋值：将 z(...) 的结果保存到 z  % 详解: 赋值：将 z(...) 的结果保存到 z

W = sum(w);  % 详解: 赋值：将 sum(...) 的结果保存到 W
Y = sum(w .* z);  % 详解: 赋值：将 sum(...) 的结果保存到 Y
YY = sum(w .* z .* z);  % 详解: 赋值：将 sum(...) 的结果保存到 YY
YTY = sum(w .* z .* z);  % 详解: 赋值：将 sum(...) 的结果保存到 YTY
X = sum(w .* x);  % 详解: 赋值：将 sum(...) 的结果保存到 X
XX = sum(w .* x .* x);  % 详解: 赋值：将 sum(...) 的结果保存到 XX
XY = sum(w .* x .* z);  % 详解: 赋值：将 sum(...) 的结果保存到 XY

[b, a] = clg_Mstep_simple(W, Y, YY, YTY, X, XX, XY);  % 详解: 执行语句
error = sum(w .* (z - (a*x + b)).^2 );  % 详解: 赋值：将 sum(...) 的结果保存到 error

if 0  % 详解: 条件判断：if (0)
  seed = 1;  % 详解: 赋值：计算表达式并保存到 seed
  rand('state', seed);   randn('state', seed);  % 详解: 调用函数：rand('state', seed); randn('state', seed)
  x = -10:10;  % 详解: 赋值：计算表达式并保存到 x
  N = length(x);  % 详解: 赋值：将 length(...) 的结果保存到 N
  noise = randn(1,N);  % 详解: 赋值：将 randn(...) 的结果保存到 noise
  aTrue = rand(1,1);  % 详解: 赋值：将 rand(...) 的结果保存到 aTrue
  bTrue = rand(1,1);  % 详解: 赋值：将 rand(...) 的结果保存到 bTrue
  z = aTrue*x + bTrue + noise;  % 详解: 赋值：计算表达式并保存到 z
  
  w = ones(1,N);  % 详解: 赋值：将 ones(...) 的结果保存到 w
  [a, b, err] = weightedRegression(x, z, w);  % 详解: 执行语句
  
  b2=regress(z(:), [x(:) ones(N,1)]);  % 详解: 赋值：将 regress(...) 的结果保存到 b2
  assert(approxeq(b,b2(2)))  % 详解: 调用函数：assert(approxeq(b,b2(2)))
  assert(approxeq(a,b2(1)))  % 详解: 调用函数：assert(approxeq(a,b2(1)))

  w(15) = 1000;  % 详解: 执行语句
  [aW, bW, errW] = weightedRegression(x, z, w);  % 详解: 执行语句

  figure;  % 详解: 执行语句
  plot(x, z, 'ro')  % 详解: 调用函数：plot(x, z, 'ro')
  hold on  % 详解: 执行语句
  plot(x, a*x+b, 'bx-')  % 详解: 调用函数：plot(x, a*x+b, 'bx-')
  plot(x, aW*x+bW, 'gs-')  % 详解: 调用函数：plot(x, aW*x+bW, 'gs-')
  title(sprintf('a=%5.2f, aHat=%5.2f, aWHat=%5.3f, b=%5.2f, bHat=%5.2f, bWHat=%5.3f, err=%5.3f, errW=%5.3f', ...  % 详解: 图形标注/图例
		aTrue, a, aW, bTrue, b, bW, err, errW))  % 详解: 执行语句
  legend('truth', 'ls', 'wls')  % 详解: 调用函数：legend('truth', 'ls', 'wls')
  
end  % 详解: 执行语句





