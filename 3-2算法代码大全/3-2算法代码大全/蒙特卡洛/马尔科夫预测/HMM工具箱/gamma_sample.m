% 文件: gamma_sample.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function r = gamrnd(a,b,m,n);  % 详解: 执行语句




if nargin < 2,  % 详解: 条件判断：if (nargin < 2,)
   error('Requires at least two input arguments.');  % 详解: 调用函数：error('Requires at least two input arguments.')
end  % 详解: 执行语句


if nargin == 2  % 详解: 条件判断：if (nargin == 2)
   [errorcode rows columns] = rndcheck(2,2,a,b);  % 详解: 执行语句
end  % 详解: 执行语句

if nargin == 3  % 详解: 条件判断：if (nargin == 3)
   [errorcode rows columns] = rndcheck(3,2,a,b,m);  % 详解: 执行语句
end  % 详解: 执行语句

if nargin == 4  % 详解: 条件判断：if (nargin == 4)
   [errorcode rows columns] = rndcheck(4,2,a,b,m,n);  % 详解: 执行语句
end  % 详解: 执行语句

if errorcode > 0  % 详解: 条件判断：if (errorcode > 0)
   error('Size information is inconsistent.');  % 详解: 调用函数：error('Size information is inconsistent.')
end  % 详解: 执行语句

lth = rows*columns;  % 详解: 赋值：计算表达式并保存到 lth
r = zeros(lth,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 r
a = a(:); b = b(:);  % 详解: 赋值：将 a(...) 的结果保存到 a

scalara = (length(a) == 1);  % 详解: 赋值：计算表达式并保存到 scalara
if scalara  % 详解: 条件判断：if (scalara)
   a = a*ones(lth,1);  % 详解: 赋值：计算表达式并保存到 a
end  % 详解: 执行语句

scalarb = (length(b) == 1);  % 详解: 赋值：计算表达式并保存到 scalarb
if scalarb  % 详解: 条件判断：if (scalarb)
   b = b*ones(lth,1);  % 详解: 赋值：计算表达式并保存到 b
end  % 详解: 执行语句

k = find(a == 1);  % 详解: 赋值：将 find(...) 的结果保存到 k
if any(k)  % 详解: 条件判断：if (any(k))
   r(k) = -b(k) .* log(rand(size(k)));  % 详解: 调用函数：r(k) = -b(k) .* log(rand(size(k)))
end  % 详解: 执行语句


k = find(a < 1 & a > 0);  % 详解: 赋值：将 find(...) 的结果保存到 k
if any(k)  % 详解: 条件判断：if (any(k))
  c = zeros(lth,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 c
  d = zeros(lth,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 d
  c(k) = 1 ./ a(k);  % 详解: 调用函数：c(k) = 1 ./ a(k)
  d(k) = 1 ./ (1 - a(k));  % 详解: 调用函数：d(k) = 1 ./ (1 - a(k))
  accept = k;  % 详解: 赋值：计算表达式并保存到 accept
  while ~isempty(accept)  % 详解: while 循环：当 (~isempty(accept)) 为真时迭代
    u = rand(size(accept));  % 详解: 赋值：将 rand(...) 的结果保存到 u
    v = rand(size(accept));  % 详解: 赋值：将 rand(...) 的结果保存到 v
    x = u .^ c(accept);  % 详解: 赋值：计算表达式并保存到 x
    y = v .^ d(accept);  % 详解: 赋值：计算表达式并保存到 y
    k1 = find((x + y) <= 1);  % 详解: 赋值：将 find(...) 的结果保存到 k1
    if ~isempty(k1)  % 详解: 条件判断：if (~isempty(k1))
      e = -log(rand(size(k1)));  % 详解: 赋值：计算表达式并保存到 e
      r(accept(k1)) = e .* x(k1) ./ (x(k1) + y(k1));  % 详解: 调用函数：r(accept(k1)) = e .* x(k1) ./ (x(k1) + y(k1))
      accept(k1) = [];  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  r(k) = r(k) .* b(k);  % 详解: 调用函数：r(k) = r(k) .* b(k)
end  % 详解: 执行语句

k = find(a > 1);  % 详解: 赋值：将 find(...) 的结果保存到 k
bb = zeros(size(a));  % 详解: 赋值：将 zeros(...) 的结果保存到 bb
c  = bb;  % 详解: 赋值：计算表达式并保存到 c
if any(k)  % 详解: 条件判断：if (any(k))
  bb(k) = a(k) - 1;  % 详解: 执行语句
  c(k) = 3 * a(k) - 3/4;  % 详解: 执行语句
  accept = k;  % 详解: 赋值：计算表达式并保存到 accept
  count = 1;  % 详解: 赋值：计算表达式并保存到 count
  while ~isempty(accept)  % 详解: while 循环：当 (~isempty(accept)) 为真时迭代
    m = length(accept);  % 详解: 赋值：将 length(...) 的结果保存到 m
    u = rand(m,1);  % 详解: 赋值：将 rand(...) 的结果保存到 u
    v = rand(m,1);  % 详解: 赋值：将 rand(...) 的结果保存到 v
    w = u .* (1 - u);  % 详解: 赋值：计算表达式并保存到 w
    y = sqrt(c(accept) ./ w) .* (u - 0.5);  % 详解: 赋值：将 sqrt(...) 的结果保存到 y
    x = bb(accept) + y;  % 详解: 赋值：将 bb(...) 的结果保存到 x
    k1 = find(x >= 0);  % 详解: 赋值：将 find(...) 的结果保存到 k1
    if ~isempty(k1)  % 详解: 条件判断：if (~isempty(k1))
      z = 64 * (w .^ 3) .* (v .^ 2);  % 详解: 赋值：计算表达式并保存到 z
      k2 = (z(k1) <= (1 - 2 * (y(k1) .^2) ./ x(k1)));  % 详解: 赋值：计算表达式并保存到 k2
      k3 = k1(find(k2));  % 详解: 赋值：将 k1(...) 的结果保存到 k3
      r(accept(k3)) = x(k3);  % 详解: 调用函数：r(accept(k3)) = x(k3)
      k4 = k1(find(~k2));  % 详解: 赋值：将 k1(...) 的结果保存到 k4
      k5 = k4(find(log(z(k4)) <= (2*(bb(accept(k4)).*log(x(k4)./bb(accept(k4)))-y(k4)))));  % 详解: 赋值：将 k4(...) 的结果保存到 k5
      r(accept(k5)) = x(k5);  % 详解: 调用函数：r(accept(k5)) = x(k5)
      omit = [k3; k5];  % 详解: 赋值：计算表达式并保存到 omit
      accept(omit) = [];  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  r(k) = r(k) .* b(k);  % 详解: 调用函数：r(k) = r(k) .* b(k)
end  % 详解: 执行语句

r(b <= 0 | a <= 0) = NaN;  % 详解: 执行语句

r = reshape(r,rows,columns);  % 详解: 赋值：将 reshape(...) 的结果保存到 r




