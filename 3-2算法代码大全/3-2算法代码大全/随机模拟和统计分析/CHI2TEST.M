% 文件: CHI2TEST.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [h,sig]=chi2test(n,x,p,alpha)  % 详解: 函数定义：chi2test(n,x,p,alpha), 返回：h,sig


if nargin<4,alpha=0.05;end  % 详解: 条件判断：if (nargin<4,alpha=0.05;end)
if min(size(x))~=1|min(size(p))~=1,  % 详解: 条件判断：if (min(size(x))~=1|min(size(p))~=1,)
   error('X,P must be vectors');  % 详解: 调用函数：error('X,P must be vectors')
end  % 详解: 执行语句
kx=length(x);kp=length(p);  % 详解: 赋值：将 length(...) 的结果保存到 kx
if kx~=kp,  % 详解: 条件判断：if (kx~=kp,)
   error('Dimension of vectors must be agree');  % 详解: 调用函数：error('Dimension of vectors must be agree')
end  % 详解: 执行语句

chi2=sum((x-p).^2./p);  % 详解: 赋值：将 sum(...) 的结果保存到 chi2
h=0;  % 详解: 赋值：计算表达式并保存到 h
sig=1-chi2cdf(chi2,kx-1);  % 详解: 赋值：计算表达式并保存到 sig
if sig<alpha, h=1;end  % 详解: 条件判断：if (sig<alpha, h=1;end)




