% 文件: QUADG.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [int, tol1,ix]= quadg(fun,xlow,xhigh,tol,trace,p1,p2,p3,p4,p5,p6,p7,p8,p9)  % 详解: 函数定义：quadg(fun,xlow,xhigh,tol,trace,p1,p2,p3,p4,p5,p6,p7,p8,p9), 返回：int, tol1,ix
global b2  % 详解: 执行语句
global w2  % 详解: 执行语句


if exist('tol')~=1,  % 详解: 条件判断：if (exist('tol')~=1,)
  tol=1e-3;  % 详解: 赋值：计算表达式并保存到 tol
elseif isempty(tol),  % 详解: 条件判断：elseif (isempty(tol),)
  tol=1e-3;  % 详解: 赋值：计算表达式并保存到 tol
end  % 详解: 执行语句
if exist('trace')~=1,  % 详解: 条件判断：if (exist('trace')~=1,)
  trace=0;  % 详解: 赋值：计算表达式并保存到 trace
elseif isempty(trace),  % 详解: 条件判断：elseif (isempty(trace),)
  trace=0;  % 详解: 赋值：计算表达式并保存到 trace
else,  % 详解: 条件判断：else 分支
  trace=1;  % 详解: 赋值：计算表达式并保存到 trace
end  % 详解: 执行语句


if prod(size(xlow))==1,  % 详解: 条件判断：if (prod(size(xlow))==1,)
  xlow=xlow(ones(size(xhigh)));;  % 详解: 赋值：将 xlow(...) 的结果保存到 xlow
elseif prod(size(xhigh))==1,  % 详解: 条件判断：elseif (prod(size(xhigh))==1,)
  xhigh=xhigh(ones(size(xlow)));;  % 详解: 赋值：将 xhigh(...) 的结果保存到 xhigh
elseif any( size(xhigh)~=size(xlow) )  % 详解: 条件判断：elseif (any( size(xhigh)~=size(xlow) ))
  error('The input must have equal size!')  % 详解: 调用函数：error('The input must have equal size!')
end  % 详解: 执行语句


if any(fun=='(')  & any(fun=='x'),  % 详解: 条件判断：if (any(fun=='(')  & any(fun=='x'),)
  exec_string=['y=',fun ';'];  % 详解: 赋值：计算表达式并保存到 exec_string
else  % 详解: 条件判断：else 分支
  exec_string=['y=',fun,'(x'];  % 详解: 赋值：计算表达式并保存到 exec_string
  num_parameters=nargin-5;  % 详解: 赋值：计算表达式并保存到 num_parameters
  for i=1:num_parameters,  % 详解: for 循环：迭代变量 i 遍历 1:num_parameters,
    xvar=['p' int2str(i)];  % 详解: 赋值：计算表达式并保存到 xvar
    if eval(['~ischar(' ,xvar,') &all(size(xlow)==size(' xvar,')) & length(',xvar,'(:)) ~=1' ]) ,  % 详解: 条件判断：if (eval(['~ischar(' ,xvar,') &all(size(xlow)==size(' xvar,')) & length(',xvar,'(:)) ~=1' ]) ,)
      eval([xvar, '=' ,xvar ,'(:);']);  % 详解: 调用函数：eval([xvar, '=' ,xvar ,'(:);'])
      exec_string=[exec_string,',' xvar '(k,ones(1,gn) )'];  % 详解: 赋值：计算表达式并保存到 exec_string
    else  % 详解: 条件判断：else 分支
      exec_string=[exec_string,',' xvar];  % 详解: 赋值：计算表达式并保存到 exec_string
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  exec_string=[exec_string,');'];  % 详解: 赋值：计算表达式并保存到 exec_string
end  % 详解: 执行语句
[N M]=size(xlow);  % 详解: 获取向量/矩阵尺寸
xlow=xlow(:);  % 详解: 赋值：将 xlow(...) 的结果保存到 xlow
jacob=(xhigh(:)-xlow(:))/2;  % 详解: 赋值：计算表达式并保存到 jacob
nk=N*M;  % 详解: 赋值：计算表达式并保存到 nk
k=1:nk;  % 详解: 赋值：计算表达式并保存到 k
if isempty(b2),  % 详解: 条件判断：if (isempty(b2),)
  [b2 w2]=grule(2);  % 详解: 执行语句
end  % 详解: 执行语句
gn=2;  % 详解: 赋值：计算表达式并保存到 gn
x=(b2(ones(nk,1),:)+1).*jacob(k,ones(1,gn))+xlow(k,ones(1,gn));  % 详解: 赋值：计算表达式并保存到 x
eval(exec_string);  % 详解: 调用函数：eval(exec_string)
int_old=sum(w2(ones(nk,1),:).*y,2).*jacob;  % 详解: 赋值：将 sum(...) 的结果保存到 int_old
int=int_old;tol1=int;  % 详解: 赋值：计算表达式并保存到 int
if trace==1,  % 详解: 条件判断：if (trace==1,)
  x_trace=x(:);  % 详解: 赋值：将 x(...) 的结果保存到 x_trace
  y_trace=y(:);  % 详解: 赋值：将 y(...) 的结果保存到 y_trace
end  % 详解: 执行语句




converge='n';  % 详解: 赋值：计算表达式并保存到 converge
for i=1:10,  % 详解: for 循环：迭代变量 i 遍历 1:10,
  gn=2^(i+1);  % 详解: 赋值：计算表达式并保存到 gn
  gntxt=int2str(gn);  % 详解: 赋值：将 int2str(...) 的结果保存到 gntxt
  eval(['global b',gntxt,' w',gntxt,';']);  % 详解: 调用函数：eval(['global b',gntxt,' w',gntxt,';'])
  if isempty(eval(['b',gntxt])) ,  % 详解: 条件判断：if (isempty(eval(['b',gntxt])) ,)
    eval(['[b',gntxt,',w',gntxt,']=grule(',gntxt,');']);  % 详解: 调用函数：eval(['[b',gntxt,',w',gntxt,']=grule(',gntxt,');'])
  end  % 详解: 执行语句
  eval(['x=(b',gntxt,'(ones(nk,1),:)+1).*jacob(k,ones(1, gn ))+xlow(k,ones(1,gn ));']);  % 详解: 调用函数：eval(['x=(b',gntxt,'(ones(nk,1),:)+1).*jacob(k,ones(1, gn ))+xlow(k,ones(1,gn ));'])
  eval(exec_string);  % 详解: 调用函数：eval(exec_string)
  eval(['int(k)=sum(w',gntxt,'(ones(nk,1),:).*y,2).*jacob(k);']);  % 详解: 调用函数：eval(['int(k)=sum(w',gntxt,'(ones(nk,1),:).*y,2).*jacob(k);'])


  if trace==1,  % 详解: 条件判断：if (trace==1,)
    x_trace=[x_trace;x(:)];  % 详解: 赋值：计算表达式并保存到 x_trace
    y_trace=[y_trace;y(:)];  % 详解: 赋值：计算表达式并保存到 y_trace
  end  % 详解: 执行语句
  tol1(k)=abs(int_old(k)-int(k));  % 详解: 调用函数：tol1(k)=abs(int_old(k)-int(k))
  k=find(tol1 > abs(tol*int)| tol1 > abs(tol));  % 详解: 赋值：将 find(...) 的结果保存到 k
   
  if any(k),  % 详解: 条件判断：if (any(k),)
      nk=length(k);  % 详解: 赋值：将 length(...) 的结果保存到 nk
  else  % 详解: 条件判断：else 分支
    converge='y';  % 详解: 赋值：计算表达式并保存到 converge
    break;  % 详解: 跳出循环：break
  end  % 详解: 执行语句
  int_old=int;  % 详解: 赋值：计算表达式并保存到 int_old
end  % 详解: 执行语句
int=reshape(int,N,M);  % 详解: 赋值：将 reshape(...) 的结果保存到 int
tol1=reshape(tol1,N,M);  % 详解: 赋值：将 reshape(...) 的结果保存到 tol1


if converge=='n',  % 详解: 条件判断：if (converge=='n',)
  disp('Integral did not converge--singularity likely')  % 详解: 调用函数：disp('Integral did not converge--singularity likely')
end  % 详解: 执行语句


if trace==1,  % 详解: 条件判断：if (trace==1,)
  plot(x_trace,y_trace,'+')  % 详解: 调用函数：plot(x_trace,y_trace,'+')
end  % 详解: 执行语句




