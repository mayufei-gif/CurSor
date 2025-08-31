% 文件: FMIN.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [xf,options] = fmin(funfcn,ax,bx,options,varargin)  % 详解: 函数定义：fmin(funfcn,ax,bx,options,varargin), 返回：xf,options




if nargin<4, options = []; end  % 详解: 条件判断：if (nargin<4, options = []; end)
options = foptions(options);  % 详解: 赋值：将 foptions(...) 的结果保存到 options
print = options(1);  % 详解: 赋值：将 options(...) 的结果保存到 print
tol = options(2);  % 详解: 赋值：将 options(...) 的结果保存到 tol
if (~options(14))  % 详解: 条件判断：if ((~options(14)))
    options(14) = 500;  % 详解: 执行语句
  end  % 详解: 执行语句
if nargin < 3  % 详解: 条件判断：if (nargin < 3)
  error('fmin requires three input arguments.')  % 详解: 调用函数：error('fmin requires three input arguments.')
end  % 详解: 执行语句

header = ' Func evals     x           f(x)         Procedure';  % 详解: 赋值：计算表达式并保存到 header
step='       initial';  % 详解: 赋值：计算表达式并保存到 step

funfcn = fcnchk(funfcn,length(varargin));  % 详解: 赋值：将 fcnchk(...) 的结果保存到 funfcn

num = 1;  % 详解: 赋值：计算表达式并保存到 num
seps = sqrt(eps);  % 详解: 赋值：将 sqrt(...) 的结果保存到 seps
c = 0.5*(3.0 - sqrt(5.0));  % 详解: 赋值：计算表达式并保存到 c
a = ax; b = bx;  % 详解: 赋值：计算表达式并保存到 a
v = a + c*(b-a);  % 详解: 赋值：计算表达式并保存到 v
w = v; xf = v;  % 详解: 赋值：计算表达式并保存到 w
d = 0.0; e = 0.0;  % 详解: 赋值：计算表达式并保存到 d
x= xf; fx = feval(funfcn,x,varargin{:});  % 详解: 赋值：计算表达式并保存到 x
fmin_data = [ 1 xf fx ];  % 详解: 赋值：计算表达式并保存到 fmin_data
if print > 0  % 详解: 条件判断：if (print > 0)
    disp(header)  % 详解: 调用函数：disp(header)
    disp([sprintf('%5.0f   %12.6g %12.6g ',fmin_data), step])  % 详解: 调用函数：disp([sprintf('%5.0f %12.6g %12.6g ',fmin_data), step])
end  % 详解: 执行语句
fv = fx; fw = fx;  % 详解: 赋值：计算表达式并保存到 fv
xm = 0.5*(a+b);  % 详解: 赋值：计算表达式并保存到 xm
tol1 = seps*abs(xf) + tol/3.0;   tol2 = 2.0*tol1;  % 详解: 赋值：计算表达式并保存到 tol1


options(10) = 0;  % 详解: 执行语句
while ( abs(xf-xm) > (tol2 - 0.5*(b-a)) )  % 详解: while 循环：当 (( abs(xf-xm) > (tol2 - 0.5*(b-a)) )) 为真时迭代

    num = num+1;  % 详解: 赋值：计算表达式并保存到 num
    gs = 1;  % 详解: 赋值：计算表达式并保存到 gs

    if abs(e) > tol1  % 详解: 条件判断：if (abs(e) > tol1)
        gs = 0;  % 详解: 赋值：计算表达式并保存到 gs
        r = (xf-w)*(fx-fv);  % 详解: 赋值：计算表达式并保存到 r
        q = (xf-v)*(fx-fw);  % 详解: 赋值：计算表达式并保存到 q
        p = (xf-v)*q-(xf-w)*r;  % 详解: 赋值：计算表达式并保存到 p
        q = 2.0*(q-r);  % 详解: 赋值：计算表达式并保存到 q
        if q > 0.0,  p = -p; end  % 详解: 条件判断：if (q > 0.0,  p = -p; end)
        q = abs(q);  % 详解: 赋值：将 abs(...) 的结果保存到 q
        r = e;  e = d;  % 详解: 赋值：计算表达式并保存到 r

        if ( (abs(p)<abs(0.5*q*r)) & (p>q*(a-xf)) & (p<q*(b-xf)) )  % 详解: 条件判断：if (( (abs(p)<abs(0.5*q*r)) & (p>q*(a-xf)) & (p<q*(b-xf)) ))

            d = p/q;  % 详解: 赋值：计算表达式并保存到 d
            x = xf+d;  % 详解: 赋值：计算表达式并保存到 x
            step = '       parabolic';  % 详解: 赋值：计算表达式并保存到 step
     
            if ((x-a) < tol2) | ((b-x) < tol2)  % 详解: 条件判断：if (((x-a) < tol2) | ((b-x) < tol2))
                si = sign(xm-xf) + ((xm-xf) == 0);  % 详解: 赋值：将 sign(...) 的结果保存到 si
                d = tol1*si;  % 详解: 赋值：计算表达式并保存到 d
            end  % 详解: 执行语句
        else  % 详解: 条件判断：else 分支
            gs=1;  % 详解: 赋值：计算表达式并保存到 gs
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    if gs  % 详解: 条件判断：if (gs)
        if xf >= xm, e = a-xf;    else, e = b-xf;  end  % 详解: 条件判断：if (xf >= xm, e = a-xf;    else, e = b-xf;  end)
            d = c*e;  % 详解: 赋值：计算表达式并保存到 d
            step = '       golden';  % 详解: 赋值：计算表达式并保存到 step
    end  % 详解: 执行语句

    si = sign(d) + (d == 0);  % 详解: 赋值：将 sign(...) 的结果保存到 si
    x = xf + si * max( abs(d), tol1 );  % 详解: 赋值：计算表达式并保存到 x
    fu = feval(funfcn,x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 fu
    fmin_data = [num x fu];  % 详解: 赋值：计算表达式并保存到 fmin_data
    if print > 0  % 详解: 条件判断：if (print > 0)
        disp([sprintf('%5.0f   %12.6g %12.6g ',fmin_data), step])  % 详解: 调用函数：disp([sprintf('%5.0f %12.6g %12.6g ',fmin_data), step])
    end  % 详解: 执行语句

    if fu <= fx  % 详解: 条件判断：if (fu <= fx)
        if x >= xf, a = xf; else, b = xf; end  % 详解: 条件判断：if (x >= xf, a = xf; else, b = xf; end)
        v = w; fv = fw;  % 详解: 赋值：计算表达式并保存到 v
        w = xf; fw = fx;  % 详解: 赋值：计算表达式并保存到 w
        xf = x; fx = fu;  % 详解: 赋值：计算表达式并保存到 xf
    else  % 详解: 条件判断：else 分支
        if x < xf, a = x; else,b = x; end  % 详解: 条件判断：if (x < xf, a = x; else,b = x; end)
        if ( (fu <= fw) | (w == xf) )  % 详解: 条件判断：if (( (fu <= fw) | (w == xf) ))
            v = w; fv = fw;  % 详解: 赋值：计算表达式并保存到 v
            w = x; fw = fu;  % 详解: 赋值：计算表达式并保存到 w
        elseif ( (fu <= fv) | (v == xf) | (v == w) )  % 详解: 条件判断：elseif (( (fu <= fv) | (v == xf) | (v == w) ))
            v = x; fv = fu;  % 详解: 赋值：计算表达式并保存到 v
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    xm = 0.5*(a+b);  % 详解: 赋值：计算表达式并保存到 xm
    tol1 = seps*abs(xf) + tol/3.0; tol2 = 2.0*tol1;  % 详解: 赋值：计算表达式并保存到 tol1
    if num > options(14)  % 详解: 条件判断：if (num > options(14))
        if options(1) >= 0  % 详解: 条件判断：if (options(1) >= 0)
        disp('Warning: Maximum number of function evaluations has been');  % 详解: 调用函数：disp('Warning: Maximum number of function evaluations has been')
        disp('         exceeded - increase options(14).')  % 详解: 调用函数：disp(' exceeded - increase options(14).')
        options(10)=num;  % 详解: 执行语句
        return  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句
options(8) = feval(funfcn,x,varargin{:});  % 详解: 调用函数：options(8) = feval(funfcn,x,varargin{:})
options(10)=num;  % 详解: 执行语句




