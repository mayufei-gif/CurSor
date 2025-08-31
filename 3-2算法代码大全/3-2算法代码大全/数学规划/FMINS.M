% 文件: FMINS.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x, options] = fmins(funfcn,x,options,grad,varargin)  % 详解: 函数定义：fmins(funfcn,x,options,grad,varargin), 返回：x, options


if nargin<3, options = []; end  % 详解: 条件判断：if (nargin<3, options = []; end)
options = foptions(options);  % 详解: 赋值：将 foptions(...) 的结果保存到 options
prnt = options(1);  % 详解: 赋值：将 options(...) 的结果保存到 prnt
tolx = options(2);  % 详解: 赋值：将 options(...) 的结果保存到 tolx
tolf = options(3);  % 详解: 赋值：将 options(...) 的结果保存到 tolf

funfcn = fcnchk(funfcn,length(varargin));  % 详解: 赋值：将 fcnchk(...) 的结果保存到 funfcn

n = prod(size(x));  % 详解: 赋值：将 prod(...) 的结果保存到 n
if (~options(14))  % 详解: 条件判断：if ((~options(14)))
   options(14) = 200*n;  % 详解: 执行语句
end  % 详解: 执行语句

rho = 1; chi = 2; psi = 0.5; sigma = 0.5;  % 详解: 赋值：计算表达式并保存到 rho
onesn = ones(1,n);  % 详解: 赋值：将 ones(...) 的结果保存到 onesn
two2np1 = 2:n+1;  % 详解: 赋值：计算表达式并保存到 two2np1
one2n = 1:n;  % 详解: 赋值：计算表达式并保存到 one2n

xin = x(:);  % 详解: 赋值：将 x(...) 的结果保存到 xin
v = zeros(n,n+1); fv = zeros(1,n+1);  % 详解: 赋值：将 zeros(...) 的结果保存到 v
v = xin;  % 详解: 赋值：计算表达式并保存到 v
x(:) = xin;  % 详解: 执行语句
fv = feval(funfcn,x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 fv

usual_delta = 0.05;  % 详解: 赋值：计算表达式并保存到 usual_delta
zero_term_delta = 0.00025;  % 详解: 赋值：计算表达式并保存到 zero_term_delta
for j = 1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
   y = xin;  % 详解: 赋值：计算表达式并保存到 y
   if y(j) ~= 0  % 详解: 条件判断：if (y(j) ~= 0)
      y(j) = (1 + usual_delta)*y(j);  % 详解: 调用函数：y(j) = (1 + usual_delta)*y(j)
   else  % 详解: 条件判断：else 分支
      y(j) = zero_term_delta;  % 详解: 执行语句
   end  % 详解: 执行语句
   v(:,j+1) = y;  % 详解: 执行语句
   x(:) = y; f = feval(funfcn,x,varargin{:});  % 详解: 调用函数：x(:) = y; f = feval(funfcn,x,varargin{:})
   fv(1,j+1) = f;  % 详解: 执行语句
end  % 详解: 执行语句

[fv,j] = sort(fv);  % 详解: 执行语句
v = v(:,j);  % 详解: 赋值：将 v(...) 的结果保存到 v

func_evals = n+1;  % 详解: 赋值：计算表达式并保存到 func_evals
if prnt > 0  % 详解: 条件判断：if (prnt > 0)
   clc  % 详解: 执行语句
   formatsave = get(0,{'format','formatspacing'});  % 详解: 赋值：将 get(...) 的结果保存到 formatsave
   format compact  % 详解: 执行语句
   format short e  % 详解: 执行语句
   disp(' ')  % 详解: 调用函数：disp(' ')
   disp('initial')  % 详解: 调用函数：disp('initial')
   v  % 详解: 执行语句
   fv  % 详解: 执行语句
   func_evals  % 详解: 执行语句
end  % 详解: 执行语句

while func_evals < options(14)  % 详解: while 循环：当 (func_evals < options(14)) 为真时迭代
   if max(max(abs(v(:,two2np1)-v(:,onesn)))) <= tolx &...  % 详解: 条件判断：if (max(max(abs(v(:,two2np1)-v(:,onesn)))) <= tolx &...)
         max(abs(fv(1)-fv(two2np1))) <= tolf  % 详解: 统计：最大/最小值
      break  % 详解: 跳出循环：break
   end  % 详解: 执行语句
   how = '';  % 详解: 赋值：计算表达式并保存到 how
   
   
   xbar = sum(v(:,one2n), 2)/n;  % 详解: 赋值：将 sum(...) 的结果保存到 xbar
   xr = (1 + rho)*xbar - rho*v(:,end);  % 详解: 赋值：计算表达式并保存到 xr
   x(:) = xr; fxr = feval(funfcn,x,varargin{:});  % 详解: 调用函数：x(:) = xr; fxr = feval(funfcn,x,varargin{:})
   func_evals = func_evals+1;  % 详解: 赋值：计算表达式并保存到 func_evals
   
   if fxr < fv(:,1)  % 详解: 条件判断：if (fxr < fv(:,1))
      xe = (1 + rho*chi)*xbar - rho*chi*v(:,end);  % 详解: 赋值：计算表达式并保存到 xe
      x(:) = xe; fxe = feval(funfcn,x,varargin{:});  % 详解: 调用函数：x(:) = xe; fxe = feval(funfcn,x,varargin{:})
      func_evals = func_evals+1;  % 详解: 赋值：计算表达式并保存到 func_evals
      if fxe < fxr  % 详解: 条件判断：if (fxe < fxr)
         v(:,end) = xe;  % 详解: 执行语句
         fv(:,end) = fxe;  % 详解: 执行语句
         how = 'expand';  % 详解: 赋值：计算表达式并保存到 how
      else  % 详解: 条件判断：else 分支
         v(:,end) = xr;  % 详解: 执行语句
         fv(:,end) = fxr;  % 详解: 执行语句
         how = 'reflect';  % 详解: 赋值：计算表达式并保存到 how
      end  % 详解: 执行语句
   else  % 详解: 条件判断：else 分支
      if fxr < fv(:,n)  % 详解: 条件判断：if (fxr < fv(:,n))
         v(:,end) = xr;  % 详解: 执行语句
         fv(:,end) = fxr;  % 详解: 执行语句
         how = 'reflect';  % 详解: 赋值：计算表达式并保存到 how
      else  % 详解: 条件判断：else 分支
         if fxr < fv(:,end)  % 详解: 条件判断：if (fxr < fv(:,end))
            xc = (1 + psi*rho)*xbar - psi*rho*v(:,end);  % 详解: 赋值：计算表达式并保存到 xc
            x(:) = xc; fxc = feval(funfcn,x,varargin{:});  % 详解: 调用函数：x(:) = xc; fxc = feval(funfcn,x,varargin{:})
            func_evals = func_evals+1;  % 详解: 赋值：计算表达式并保存到 func_evals
            
            if fxc <= fxr  % 详解: 条件判断：if (fxc <= fxr)
               v(:,end) = xc;  % 详解: 执行语句
               fv(:,end) = fxc;  % 详解: 执行语句
               how = 'contract outside';  % 详解: 赋值：计算表达式并保存到 how
            else  % 详解: 条件判断：else 分支
               how = 'shrink';  % 详解: 赋值：计算表达式并保存到 how
            end  % 详解: 执行语句
         else  % 详解: 条件判断：else 分支
            xcc = (1-psi)*xbar + psi*v(:,end);  % 详解: 赋值：计算表达式并保存到 xcc
            x(:) = xcc; fxcc = feval(funfcn,x,varargin{:});  % 详解: 调用函数：x(:) = xcc; fxcc = feval(funfcn,x,varargin{:})
            func_evals = func_evals+1;  % 详解: 赋值：计算表达式并保存到 func_evals
            
            if fxcc < fv(:,end)  % 详解: 条件判断：if (fxcc < fv(:,end))
               v(:,end) = xcc;  % 详解: 执行语句
               fv(:,end) = fxcc;  % 详解: 执行语句
               how = 'contract inside';  % 详解: 赋值：计算表达式并保存到 how
            else  % 详解: 条件判断：else 分支
               how = 'shrink';  % 详解: 赋值：计算表达式并保存到 how
            end  % 详解: 执行语句
         end  % 详解: 执行语句
         if strcmp(how,'shrink')  % 详解: 条件判断：if (strcmp(how,'shrink'))
            for j=two2np1  % 详解: for 循环：迭代变量 j 遍历 two2np1
               v(:,j)=v(:,1)+sigma*(v(:,j) - v(:,1));  % 详解: 调用函数：v(:,j)=v(:,1)+sigma*(v(:,j) - v(:,1))
               x(:) = v(:,j); fxcc = feval(funfcn,x,varargin{:});  % 详解: 调用函数：x(:) = v(:,j); fxcc = feval(funfcn,x,varargin{:})
            end  % 详解: 执行语句
            func_evals = func_evals + n;  % 详解: 赋值：计算表达式并保存到 func_evals
         end  % 详解: 执行语句
      end  % 详解: 执行语句
   end  % 详解: 执行语句
   [fv,j] = sort(fv);  % 详解: 执行语句
   v = v(:,j);  % 详解: 赋值：将 v(...) 的结果保存到 v
   if prnt > 0  % 详解: 条件判断：if (prnt > 0)
      disp(' ')  % 详解: 调用函数：disp(' ')
      disp(how)  % 详解: 调用函数：disp(how)
      v  % 详解: 执行语句
      fv  % 详解: 执行语句
      func_evals  % 详解: 执行语句
   end  % 详解: 执行语句
end  % 详解: 执行语句


x(:) = v(:,1);  % 详解: 调用函数：x(:) = v(:,1)
if prnt > 0,  % 详解: 条件判断：if (prnt > 0,)
   set(0,{'format','formatspacing'},formatsave);  % 详解: 调用函数：set(0,{'format','formatspacing'},formatsave)
end  % 详解: 执行语句
options(10)=func_evals;  % 详解: 执行语句
options(8)=min(fv);  % 详解: 调用函数：options(8)=min(fv)
if func_evals >= options(14)  % 详解: 条件判断：if (func_evals >= options(14))
   if options(1) >= 0  % 详解: 条件判断：if (options(1) >= 0)
      disp(' ')  % 详解: 调用函数：disp(' ')
      disp(['Maximum number of function evaluations (', ...  % 详解: 打印/显示输出
            int2str(options(14)),') has been exceeded']);  % 详解: 调用函数：int2str(options(14)),') has been exceeded'])
      disp( '         (increase OPTIONS(14)).')  % 详解: 调用函数：disp(' (increase OPTIONS(14)).')
   end  % 详解: 执行语句
end  % 详解: 执行语句








