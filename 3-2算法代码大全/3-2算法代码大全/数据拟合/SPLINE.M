% 文件: SPLINE.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function output = spline(x,y,xx)  % 详解: 执行语句



output=[];  % 详解: 赋值：计算表达式并保存到 output
n=length(x);  % 详解: 赋值：将 length(...) 的结果保存到 n
if n<2, error('There should be at least two data points.'), end  % 详解: 条件判断：if (n<2, error('There should be at least two data points.'), end)

if any(diff(x)<0), [x,ind]=sort(x); else, ind=1:n; end  % 详解: 条件判断：if (any(diff(x)<0), [x,ind]=sort(x); else, ind=1:n; end)

x=x(:); dx = diff(x);  % 详解: 赋值：将 x(...) 的结果保存到 x
if all(dx)==0, error('The data abscissae should be distinct.'), end  % 详解: 条件判断：if (all(dx)==0, error('The data abscissae should be distinct.'), end)

[yd,yn] = size(y);  % 详解: 获取向量/矩阵尺寸
if yn==1, yn=yd; y=reshape(y,1,yn); yd=1; end  % 详解: 条件判断：if (yn==1, yn=yd; y=reshape(y,1,yn); yd=1; end)

if yn==n  % 详解: 条件判断：if (yn==n)
   notaknot = 1;  % 详解: 赋值：计算表达式并保存到 notaknot
elseif yn==n+2  % 详解: 条件判断：elseif (yn==n+2)
   notaknot = 0; endslopes = y(:,[1 n+2]).'; y(:,[1 n+2])=[];  % 赋值：设置变量 notaknot  % 详解: 赋值：计算表达式并保存到 notaknot  % 详解: 赋值：计算表达式并保存到 notaknot
else  % 详解: 条件判断：else 分支
   error('Abscissa and ordinate vector should be of the same length.')  % 详解: 调用函数：error('Abscissa and ordinate vector should be of the same length.')
end  % 详解: 执行语句

yi=y(:,ind).'; dd = ones(1,yd);  % 创建全 1 矩阵/数组  % 详解: 赋值：将 y(...) 的结果保存到 yi  % 详解: 赋值：将 y(...) 的结果保存到 yi
dx = diff(x); divdif = diff(yi)./dx(:,dd);  % 详解: 赋值：将 diff(...) 的结果保存到 dx
if n==2  % 详解: 条件判断：if (n==2)
   if notaknot,  % 详解: 条件判断：if (notaknot,)
      pp=mkpp(x.',[divdif.' yi(1,:).'],yd);  % 赋值：设置变量 pp  % 详解: 赋值：将 mkpp(...) 的结果保存到 pp  % 详解: 赋值：将 mkpp(...) 的结果保存到 pp
   else  % 详解: 条件判断：else 分支
      divdif2 = diff([endslopes(1,:);divdif;endslopes(2,:)])./dx([1 1],dd);  % 详解: 赋值：将 diff(...) 的结果保存到 divdif2
      pp = mkpp(x,...  % 详解: 赋值：将 mkpp(...) 的结果保存到 pp
      [(diff(divdif2)./dx(1,dd)).' ([2 -1]*divdif2).' ...  % 详解: 执行语句
                                           endslopes(1,:).' yi(1,:).'],yd);  % 详解: 调用函数：endslopes(1,:).' yi(1,:).'],yd)
   end  % 详解: 执行语句
elseif n==3&notaknot,  % 详解: 条件判断：elseif (n==3&notaknot,)
   yi(2:3,:)=divdif;  % 详解: 执行语句
   yi(3,:)=diff(divdif)/(x(3)-x(1));  % 详解: 调用函数：yi(3,:)=diff(divdif)/(x(3)-x(1))
   yi(2,:)=yi(2,:)-yi(3,:)*dx(1);  % 详解: 调用函数：yi(2,:)=yi(2,:)-yi(3,:)*dx(1)
   pp = mkpp([x(1),x(3)],yi([3 2 1],:).',yd);  % 赋值：设置变量 pp  % 详解: 赋值：将 mkpp(...) 的结果保存到 pp  % 详解: 赋值：将 mkpp(...) 的结果保存到 pp
else  % 详解: 条件判断：else 分支
   b=zeros(n,yd);  % 详解: 赋值：将 zeros(...) 的结果保存到 b
   b(2:n-1,:)=3*(dx(2:n-1,dd).*divdif(1:n-2,:)+dx(1:n-2,dd).*divdif(2:n-1,:));  % 详解: 调用函数：b(2:n-1,:)=3*(dx(2:n-1,dd).*divdif(1:n-2,:)+dx(1:n-2,dd).*divdif(2:n-1,:))
   if notaknot  % 详解: 条件判断：if (notaknot)
      x31=x(3)-x(1);xn=x(n)-x(n-2);  % 详解: 赋值：将 x(...) 的结果保存到 x31
      b(1,:)=((dx(1)+2*x31)*dx(2)*divdif(1,:)+dx(1)^2*divdif(2,:))/x31;  % 详解: 执行语句
      b(n,:)=...  % 详解: 执行语句
      (dx(n-1)^2*divdif(n-2,:)+(2*xn+dx(n-1))*dx(n-2)*divdif(n-1,:))/xn;  % 详解: 执行语句
   else  % 详解: 条件判断：else 分支
      x31 = 0; xn = 0; b([1 n],:) = dx([1 n-2],dd).*endslopes;  % 详解: 赋值：计算表达式并保存到 x31
   end  % 详解: 执行语句
   c = spdiags([ [dx(2:n-1);xn;0] ...  % 详解: 赋值：将 spdiags(...) 的结果保存到 c
        [dx(2);2*[dx(2:n-1)+dx(1:n-2)];dx(n-2)] ...  % 详解: 执行语句
        [0;x31;dx(1:n-2)] ],[-1 0 1],n,n);  % 详解: 执行语句

   mmdflag = spparms('autommd');  % 详解: 赋值：将 spparms(...) 的结果保存到 mmdflag
   spparms('autommd',0);  % 详解: 调用函数：spparms('autommd',0)
   s=c\b;  % 详解: 赋值：计算表达式并保存到 s
   spparms('autommd',mmdflag);  % 详解: 调用函数：spparms('autommd',mmdflag)
   c4=(s(1:n-1,:)+s(2:n,:)-2*divdif(1:n-1,:))./dx(:,dd);  % 详解: 赋值：计算表达式并保存到 c4
   c3=(divdif(1:n-1,:)-s(1:n-1,:))./dx(:,dd) - c4;  % 详解: 赋值：计算表达式并保存到 c3
   pp=mkpp(x.',...  % 赋值：设置变量 pp  % 详解: 赋值：将 mkpp(...) 的结果保存到 pp  % 详解: 赋值：将 mkpp(...) 的结果保存到 pp
     reshape([(c4./dx(:,dd)).' c3.' s(1:n-1,:).' yi(1:n-1,:).'],(n-1)*yd,4),yd);  % 详解: 调用函数：reshape([(c4./dx(:,dd)).' c3.' s(1:n-1,:).' yi(1:n-1,:).'],(n-1)*yd,4),yd)
end  % 详解: 执行语句
if nargin==2  % 详解: 条件判断：if (nargin==2)
   output=pp;  % 详解: 赋值：计算表达式并保存到 output
else  % 详解: 条件判断：else 分支
   output=ppval(pp,xx);  % 详解: 赋值：将 ppval(...) 的结果保存到 output
end  % 详解: 执行语句




