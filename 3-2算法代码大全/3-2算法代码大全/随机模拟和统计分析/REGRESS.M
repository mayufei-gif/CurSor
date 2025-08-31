% 文件: REGRESS.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [b,bint,r,rint,stats] = regress(y,X,alpha)  % 详解: 函数定义：regress(y,X,alpha), 返回：b,bint,r,rint,stats



if  nargin < 2,  % 详解: 条件判断：if (nargin < 2,)
    error('REGRESS requires at least two input arguments.');  % 详解: 调用函数：error('REGRESS requires at least two input arguments.')
end  % 详解: 执行语句

if nargin == 2,  % 详解: 条件判断：if (nargin == 2,)
    alpha = 0.05;  % 详解: 赋值：计算表达式并保存到 alpha
end  % 详解: 执行语句

[n,p] = size(X);  % 详解: 获取向量/矩阵尺寸
[n1,collhs] = size(y);  % 详解: 获取向量/矩阵尺寸
if n ~= n1,  % 详解: 条件判断：if (n ~= n1,)
    error('The number of rows in Y must equal the number of rows in X.');  % 详解: 调用函数：error('The number of rows in Y must equal the number of rows in X.')
end  % 详解: 执行语句

if collhs ~= 1,  % 详解: 条件判断：if (collhs ~= 1,)
    error('Y must be a vector, not a matrix');  % 详解: 调用函数：error('Y must be a vector, not a matrix')
end  % 详解: 执行语句

[Q, R]=qr(X,0);  % 详解: 执行语句
b = R\(Q'*y);  % 赋值：设置变量 b  % 详解: 赋值：计算表达式并保存到 b  % 详解: 赋值：计算表达式并保存到 b


RI = R\eye(p);  % 详解: 赋值：计算表达式并保存到 RI
xdiag=sqrt(sum((RI .* RI)'))';  % 详解: 赋值：将 sqrt(...) 的结果保存到 xdiag
nu = n-p;  % 详解: 赋值：计算表达式并保存到 nu
yhat = X*b;  % 详解: 赋值：计算表达式并保存到 yhat
r = y-yhat;  % 详解: 赋值：计算表达式并保存到 r
if nu ~= 0  % 详解: 条件判断：if (nu ~= 0)
   rmse = norm(r)/sqrt(nu);  % 详解: 赋值：将 norm(...) 的结果保存到 rmse
else  % 详解: 条件判断：else 分支
   rmse = Inf;  % 详解: 赋值：计算表达式并保存到 rmse
end  % 详解: 执行语句
s2 = rmse^2;  % 详解: 赋值：计算表达式并保存到 s2
tval = tinv((1-alpha/2),nu);  % 详解: 赋值：将 tinv(...) 的结果保存到 tval
bint = [b-tval*xdiag*rmse, b+tval*xdiag*rmse];  % 详解: 赋值：计算表达式并保存到 bint

if nargout==5,  % 详解: 条件判断：if (nargout==5,)
   RSS = norm(yhat-mean(y))^2;  % 详解: 赋值：将 norm(...) 的结果保存到 RSS
   TSS = norm(y-mean(y))^2;  % 详解: 赋值：将 norm(...) 的结果保存到 TSS
   r2 = RSS/TSS;  % 详解: 赋值：计算表达式并保存到 r2
   F = (RSS/(p-1))/s2;  % 详解: 赋值：计算表达式并保存到 F
   prob = 1 - fcdf(F,p-1,nu);  % 详解: 赋值：计算表达式并保存到 prob
   stats = [r2 F prob];  % 详解: 赋值：计算表达式并保存到 stats
end  % 详解: 执行语句

T = X*RI;  % 详解: 赋值：计算表达式并保存到 T
hatdiag=sum((T .* T)')';  % 详解: 赋值：将 sum(...) 的结果保存到 hatdiag
if nu < 1,  % 详解: 条件判断：if (nu < 1,)
  ser=rmse*ones(length(y),1);  % 详解: 赋值：计算表达式并保存到 ser
elseif nu > 1  % 详解: 条件判断：elseif (nu > 1)
  sigmai = sqrt((nu*s2/(nu-1)) - (r .^2 ./ ((nu-1) .* (1-hatdiag))));  % 详解: 赋值：将 sqrt(...) 的结果保存到 sigmai
  ser = sqrt(1-hatdiag) .* sigmai;  % 详解: 赋值：将 sqrt(...) 的结果保存到 ser
elseif nu == 1  % 详解: 条件判断：elseif (nu == 1)
  ser = sqrt(1-hatdiag) .* rmse;  % 详解: 赋值：将 sqrt(...) 的结果保存到 ser
end  % 详解: 执行语句

ti= r ./ ser;  % 详解: 赋值：计算表达式并保存到 ti

Z=[(r-tval*ser) (r+tval*ser)]';  % 赋值：设置变量 Z  % 详解: 赋值：计算表达式并保存到 Z  % 详解: 赋值：计算表达式并保存到 Z
rint=Z';  % 赋值：设置变量 rint  % 详解: 赋值：计算表达式并保存到 rint  % 详解: 赋值：计算表达式并保存到 rint




