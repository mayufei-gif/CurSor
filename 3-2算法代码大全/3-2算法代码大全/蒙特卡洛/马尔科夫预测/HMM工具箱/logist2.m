% 文件: logist2.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [beta,p,lli] = logist2(y,x,w)  % 详解: 函数定义：logist2(y,x,w), 返回：beta,p,lli



error(nargchk(2,3,nargin));  % 详解: 调用函数：error(nargchk(2,3,nargin))

if size(y,2) ~= 1,  % 详解: 条件判断：if (size(y,2) ~= 1,)
  error('Input y not a column vector.');  % 详解: 调用函数：error('Input y not a column vector.')
end  % 详解: 执行语句
if size(y,1) ~= size(x,1),  % 详解: 条件判断：if (size(y,1) ~= size(x,1),)
  error('Input x,y sizes mismatched.');  % 详解: 调用函数：error('Input x,y sizes mismatched.')
end  % 详解: 执行语句

[N,k] = size(x);  % 详解: 获取向量/矩阵尺寸

if nargin < 3,  % 详解: 条件判断：if (nargin < 3,)
  w = 1;  % 详解: 赋值：计算表达式并保存到 w
end  % 详解: 执行语句

w = w / max(w);  % 详解: 赋值：计算表达式并保存到 w

beta = zeros(k,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 beta

iter = 0;  % 详解: 赋值：计算表达式并保存到 iter
lli = 0;  % 详解: 赋值：计算表达式并保存到 lli
while 1==1,  % 详解: while 循环：当 (1==1,) 为真时迭代
  iter = iter + 1;  % 详解: 赋值：计算表达式并保存到 iter
  
  p = 1 ./ (1 + exp(-x*beta));  % 详解: 赋值：计算表达式并保存到 p
  
  lli_prev = lli;  % 详解: 赋值：计算表达式并保存到 lli_prev
  lli = sum( w .* (y.*log(p+eps) + (1-y).*log(1-p+eps)) );  % 详解: 赋值：将 sum(...) 的结果保存到 lli

  wt = w .* p .* (1-p);  % 详解: 赋值：计算表达式并保存到 wt

  deriv = x'*(w.*(y-p));  % 赋值：设置变量 deriv  % 详解: 赋值：计算表达式并保存到 deriv  % 详解: 赋值：计算表达式并保存到 deriv

  hess = zeros(k,k);  % 详解: 赋值：将 zeros(...) 的结果保存到 hess
  for i = 1:k,  % 详解: for 循环：迭代变量 i 遍历 1:k,
    wxi = wt .* x(:,i);  % 详解: 赋值：计算表达式并保存到 wxi
    for j = i:k,  % 详解: for 循环：迭代变量 j 遍历 i:k,
      hij = wxi' * x(:,j);  % 赋值：设置变量 hij  % 详解: 赋值：计算表达式并保存到 hij  % 详解: 赋值：计算表达式并保存到 hij
      hess(i,j) = -hij;  % 详解: 执行语句
      hess(j,i) = -hij;  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句

  if (rcond(hess) < eps),  % 详解: 条件判断：if ((rcond(hess) < eps),)
    error(['Stopped at iteration ' num2str(iter) ...  % 详解: 执行语句
           ' because Hessian is poorly conditioned.']);  % 详解: 执行语句
    break;  % 详解: 跳出循环：break
  end;  % 详解: 执行语句

  step = hess\deriv;  % 详解: 赋值：计算表达式并保存到 step
  beta = beta - step;  % 详解: 赋值：计算表达式并保存到 beta

  tol = 1e-6;  % 详解: 赋值：计算表达式并保存到 tol
  if abs(deriv'*step/k) < tol, break; end;  % 条件判断：if 分支开始  % 详解: 条件判断：if (abs(deriv'*step/k) < tol, break; end;)  % 详解: 条件判断：if (abs(deriv'*step/k) < tol, break; end;  % 条件判断：if 分支开始  % 详解: 条件判断：if (abs(deriv'*step/k) < tol, break; end;))

end;  % 详解: 执行语句





