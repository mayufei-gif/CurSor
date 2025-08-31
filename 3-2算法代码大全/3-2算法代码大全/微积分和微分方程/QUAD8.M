% 文件: QUAD8.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [Q,cnt] = quad8(funfcn,a,b,tol,trace,varargin)  % 详解: 函数定义：quad8(funfcn,a,b,tol,trace,varargin), 返回：Q,cnt



if nargin < 4, tol = [1.e-3 0]; trace = 0; end  % 详解: 条件判断：if (nargin < 4, tol = [1.e-3 0]; trace = 0; end)
if nargin < 5, trace = 0; end  % 详解: 条件判断：if (nargin < 5, trace = 0; end)
if isempty(tol), tol = [1.e-3 0]; end  % 详解: 条件判断：if (isempty(tol), tol = [1.e-3 0]; end)
if length(tol)==1, tol = [tol 0]; end  % 详解: 条件判断：if (length(tol)==1, tol = [tol 0]; end)
if isempty(trace), trace = 0; end  % 详解: 条件判断：if (isempty(trace), trace = 0; end)
h = b - a;  % 详解: 赋值：计算表达式并保存到 h

w = [3956 23552 -3712 41984 -18160 41984 -3712 23552 3956]/14175;  % 详解: 赋值：计算表达式并保存到 w
x = a + (0:8)*(b-a)/8;  % 详解: 赋值：计算表达式并保存到 x
y = feval(funfcn,x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 y

yllow = min([min(real(y)) min(imag(y))]);  % 详解: 赋值：将 min(...) 的结果保存到 yllow
ylhi = max([max(real(y)) max(imag(y))]);  % 详解: 赋值：将 max(...) 的结果保存到 ylhi
lims = [min(x) max(x) yllow ylhi];  % 详解: 赋值：计算表达式并保存到 lims
ind = find(~isfinite(lims));  % 详解: 赋值：将 find(...) 的结果保存到 ind
if ~isempty(ind)  % 详解: 条件判断：if (~isempty(ind))
    [mind,nind] = size(ind);  % 详解: 获取向量/矩阵尺寸
    lims(ind) = 1.e30*(-ones(mind,nind) .^ rem(ind,2));  % 详解: 调用函数：lims(ind) = 1.e30*(-ones(mind,nind) .^ rem(ind,2))
end  % 详解: 执行语句
if trace  % 详解: 条件判断：if (trace)
    axis(lims);  % 详解: 调用函数：axis(lims)
    plot([a b],[real(y(1)) real(y(9))],'.'), hold on  % 详解: 绘图：二维曲线
    if ~isreal(y)  % 详解: 条件判断：if (~isreal(y))
        plot([a b],[imag(y(1)) imag(y(9))],'+')  % 详解: 调用函数：plot([a b],[imag(y(1)) imag(y(9))],'+')
    end  % 详解: 执行语句
    drawnow  % 详解: 执行语句
end  % 详解: 执行语句
lev = 1;  % 详解: 赋值：计算表达式并保存到 lev

if ~isreal(y)  % 详解: 条件判断：if (~isreal(y))
    Q0 = 1e30;  % 详解: 赋值：计算表达式并保存到 Q0
else  % 详解: 条件判断：else 分支
    Q0 = inf;  % 详解: 赋值：计算表达式并保存到 Q0
end  % 详解: 执行语句
recur_lev_excess = 0;  % 详解: 赋值：计算表达式并保存到 recur_lev_excess
[Q,cnt,recur_lev_excess] = quad8stp(funfcn,a,b,tol,lev,w,x,y,Q0,trace,recur_lev_excess,varargin{:});  % 详解: 执行语句
cnt = cnt + 9;  % 详解: 赋值：计算表达式并保存到 cnt
if trace  % 详解: 条件判断：if (trace)
    hold off  % 详解: 执行语句
    axis('auto');  % 详解: 调用函数：axis('auto')
end  % 详解: 执行语句
if (recur_lev_excess > 1)  % 详解: 条件判断：if ((recur_lev_excess > 1))
    warning(sprintf('Recursion level limit reached %d times.', ...  % 详解: 执行语句
         recur_lev_excess ))  % 详解: 执行语句
end  % 详解: 执行语句

function [Q,cnt,recur_lev_excess] = quad8stp(FunFcn,a,b,tol,lev,w,x0,f0,Q0,trace,recur_lev_excess,varargin)  % 详解: 函数定义：quad8stp(FunFcn,a,b,tol,lev,w,x0,f0,Q0,trace,recur_lev_excess,varargin), 返回：Q,cnt,recur_lev_excess


LEVMAX = 10;  % 详解: 赋值：计算表达式并保存到 LEVMAX

x = zeros(1,17);  % 详解: 赋值：将 zeros(...) 的结果保存到 x
f = zeros(1,17);  % 详解: 赋值：将 zeros(...) 的结果保存到 f
x(1:2:17) = x0;  % 详解: 执行语句
f(1:2:17) = f0;  % 详解: 执行语句
x(2:2:16) = (x0(1:8) + x0(2:9))/2;  % 详解: 执行语句
f(2:2:16) = feval(FunFcn,x(2:2:16),varargin{:});  % 详解: 调用函数：f(2:2:16) = feval(FunFcn,x(2:2:16),varargin{:})
if trace  % 详解: 条件判断：if (trace)
    plot(x(2:2:16),f(2:2:16),'.');  % 详解: 调用函数：plot(x(2:2:16),f(2:2:16),'.')
    if ~isreal(f)  % 详解: 条件判断：if (~isreal(f))
        plot(x(2:2:16),imag(f(2:2:16)),'+');  % 详解: 调用函数：plot(x(2:2:16),imag(f(2:2:16)),'+')
    end  % 详解: 执行语句
    drawnow  % 详解: 执行语句
end  % 详解: 执行语句
cnt = 8;  % 详解: 赋值：计算表达式并保存到 cnt

h = (b-a)/16;  % 详解: 赋值：计算表达式并保存到 h
Q1 = h*w*f(1:9).';  % 赋值：设置变量 Q1  % 详解: 赋值：计算表达式并保存到 Q1  % 详解: 赋值：计算表达式并保存到 Q1
Q2 = h*w*f(9:17).';  % 赋值：设置变量 Q2  % 详解: 赋值：计算表达式并保存到 Q2  % 详解: 赋值：计算表达式并保存到 Q2
Q = Q1 + Q2;  % 详解: 赋值：计算表达式并保存到 Q

if (abs(Q - Q0) > tol(1)*abs(Q)+tol(2)) & (lev <= LEVMAX)  % 详解: 条件判断：if ((abs(Q - Q0) > tol(1)*abs(Q)+tol(2)) & (lev <= LEVMAX))
   c = (a+b)/2;  % 详解: 赋值：计算表达式并保存到 c
   [Q1,cnt1,recur_lev_excess] = quad8stp(FunFcn,a,c,tol/2,lev+1,w,x(1:9),f(1:9),Q1,trace,recur_lev_excess,varargin{:});  % 详解: 执行语句
   [Q2,cnt2,recur_lev_excess] = quad8stp(FunFcn,c,b,tol/2,lev+1,w,x(9:17),f(9:17),Q2,trace,recur_lev_excess,varargin{:});  % 详解: 执行语句
   Q = Q1 + Q2;  % 详解: 赋值：计算表达式并保存到 Q
   cnt = cnt + cnt1 + cnt2;  % 详解: 赋值：计算表达式并保存到 cnt
elseif (lev > LEVMAX)  % 详解: 条件判断：elseif ((lev > LEVMAX))
   if ~recur_lev_excess  % 详解: 条件判断：if (~recur_lev_excess)
      warning('Recursion level limit reached in quad8. Singularity likely.')  % 详解: 调用函数：warning('Recursion level limit reached in quad8. Singularity likely.')
      recur_lev_excess = 1;  % 详解: 赋值：计算表达式并保存到 recur_lev_excess
   else  % 详解: 条件判断：else 分支
      recur_lev_excess = recur_lev_excess + 1;  % 详解: 赋值：计算表达式并保存到 recur_lev_excess
   end  % 详解: 执行语句
end  % 详解: 执行语句




