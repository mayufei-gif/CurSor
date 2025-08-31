% 文件: ODE45.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [tout,yout,varargout] = ode45(odefile,tspan,y0,options,varargin)  % 详解: 函数定义：ode45(odefile,tspan,y0,options,varargin), 返回：tout,yout,varargout




true = 1;  % 详解: 赋值：计算表达式并保存到 true
false = ~true;  % 详解: 赋值：计算表达式并保存到 false

nsteps = 0;  % 详解: 赋值：计算表达式并保存到 nsteps
nfailed = 0;  % 详解: 赋值：计算表达式并保存到 nfailed
nfevals = 0;  % 详解: 赋值：计算表达式并保存到 nfevals
npds = 0;  % 详解: 赋值：计算表达式并保存到 npds
ndecomps = 0;  % 详解: 赋值：计算表达式并保存到 ndecomps
nsolves = 0;  % 详解: 赋值：计算表达式并保存到 nsolves

if nargin == 0  % 详解: 条件判断：if (nargin == 0)
  error('Not enough input arguments.  See ODE45.');  % 详解: 调用函数：error('Not enough input arguments. See ODE45.')
elseif ~isstr(odefile) & ~isa(odefile, 'inline')  % 详解: 条件判断：elseif (~isstr(odefile) & ~isa(odefile, 'inline'))
  error('First argument must be a single-quoted string.  See ODE45.');  % 详解: 调用函数：error('First argument must be a single-quoted string. See ODE45.')
end  % 详解: 执行语句

if nargin == 1  % 详解: 条件判断：if (nargin == 1)
  tspan = []; y0 = []; options = [];  % 详解: 赋值：计算表达式并保存到 tspan
elseif nargin == 2  % 详解: 条件判断：elseif (nargin == 2)
  y0 = []; options = [];  % 详解: 赋值：计算表达式并保存到 y0
elseif nargin == 3  % 详解: 条件判断：elseif (nargin == 3)
  options = [];  % 详解: 赋值：计算表达式并保存到 options
elseif ~isempty(options) & ~isa(options,'struct')  % 详解: 条件判断：elseif (~isempty(options) & ~isa(options,'struct'))
  if (length(tspan) == 1) & (length(y0) == 1) & (min(size(options)) == 1)  % 详解: 条件判断：if ((length(tspan) == 1) & (length(y0) == 1) & (min(size(options)) == 1))
    tspan = [tspan; y0];  % 详解: 赋值：计算表达式并保存到 tspan
    y0 = options;  % 详解: 赋值：计算表达式并保存到 y0
    options = [];  % 详解: 赋值：计算表达式并保存到 options
    varargin = {};  % 详解: 赋值：计算表达式并保存到 varargin
    msg = sprintf('Use ode45(''%s'',tspan,y0,...) instead.',odefile);  % 详解: 赋值：将 sprintf(...) 的结果保存到 msg
    warning(['Obsolete syntax.  ' msg]);  % 详解: 调用函数：warning(['Obsolete syntax. ' msg])
  else  % 详解: 条件判断：else 分支
    error('Correct syntax is ode45(''odefile'',tspan,y0,options).');  % 详解: 调用函数：error('Correct syntax is ode45(''odefile'',tspan,y0,options).')
  end  % 详解: 执行语句
end  % 详解: 执行语句

if isempty(tspan) | isempty(y0)  % 详解: 条件判断：if (isempty(tspan) | isempty(y0))
  if (nargout(odefile) < 3) & (nargout(odefile) ~= -1)  % 详解: 条件判断：if ((nargout(odefile) < 3) & (nargout(odefile) ~= -1))
    msg = sprintf('Use ode45(''%s'',tspan,y0,...) instead.',odefile);  % 详解: 赋值：将 sprintf(...) 的结果保存到 msg
    error(['No default parameters in ' upper(odefile) '.  ' msg]);  % 详解: 调用函数：error(['No default parameters in ' upper(odefile) '. ' msg])
  end  % 详解: 执行语句
  [def_tspan,def_y0,def_options] = feval(odefile,[],[],'init',varargin{:});  % 详解: 执行语句
  if isempty(tspan)  % 详解: 条件判断：if (isempty(tspan))
    tspan = def_tspan;  % 详解: 赋值：计算表达式并保存到 tspan
  end  % 详解: 执行语句
  if isempty(y0)  % 详解: 条件判断：if (isempty(y0))
    y0 = def_y0;  % 详解: 赋值：计算表达式并保存到 y0
  end  % 详解: 执行语句
  if isempty(options)  % 详解: 条件判断：if (isempty(options))
    options = def_options;  % 详解: 赋值：计算表达式并保存到 options
  else  % 详解: 条件判断：else 分支
    options = odeset(def_options,options);  % 详解: 赋值：将 odeset(...) 的结果保存到 options
  end  % 详解: 执行语句
end  % 详解: 执行语句

tspan = tspan(:);  % 详解: 赋值：将 tspan(...) 的结果保存到 tspan
ntspan = length(tspan);  % 详解: 赋值：将 length(...) 的结果保存到 ntspan
if ntspan == 1  % 详解: 条件判断：if (ntspan == 1)
  t0 = 0;  % 详解: 赋值：计算表达式并保存到 t0
  next = 1;  % 详解: 赋值：计算表达式并保存到 next
else  % 详解: 条件判断：else 分支
  t0 = tspan(1);  % 详解: 赋值：将 tspan(...) 的结果保存到 t0
  next = 2;  % 详解: 赋值：计算表达式并保存到 next
end  % 详解: 执行语句
tfinal = tspan(ntspan);  % 详解: 赋值：将 tspan(...) 的结果保存到 tfinal
if t0 == tfinal  % 详解: 条件判断：if (t0 == tfinal)
  error('The last entry in tspan must be different from the first entry.');  % 详解: 调用函数：error('The last entry in tspan must be different from the first entry.')
end  % 详解: 执行语句
tdir = sign(tfinal - t0);  % 详解: 赋值：将 sign(...) 的结果保存到 tdir
if any(tdir * (tspan(2:ntspan) - tspan(1:ntspan-1)) <= 0)  % 详解: 条件判断：if (any(tdir * (tspan(2:ntspan) - tspan(1:ntspan-1)) <= 0))
  error('The entries in tspan must strictly increase or decrease.');  % 详解: 调用函数：error('The entries in tspan must strictly increase or decrease.')
end  % 详解: 执行语句

t = t0;  % 详解: 赋值：计算表达式并保存到 t
y = y0(:);  % 详解: 赋值：将 y0(...) 的结果保存到 y
neq = length(y);  % 详解: 赋值：将 length(...) 的结果保存到 neq

rtol = odeget(options,'RelTol',1e-3);  % 详解: 赋值：将 odeget(...) 的结果保存到 rtol
if (length(rtol) ~= 1) | (rtol <= 0)  % 详解: 条件判断：if ((length(rtol) ~= 1) | (rtol <= 0))
  error('RelTol must be a positive scalar.');  % 详解: 调用函数：error('RelTol must be a positive scalar.')
end  % 详解: 执行语句
if rtol < 100 * eps  % 详解: 条件判断：if (rtol < 100 * eps)
  rtol = 100 * eps;  % 详解: 赋值：计算表达式并保存到 rtol
  warning(['RelTol has been increased to ' num2str(rtol) '.']);  % 详解: 调用函数：warning(['RelTol has been increased to ' num2str(rtol) '.'])
end  % 详解: 执行语句

atol = odeget(options,'AbsTol',1e-6);  % 详解: 赋值：将 odeget(...) 的结果保存到 atol
if any(atol <= 0)  % 详解: 条件判断：if (any(atol <= 0))
  error('AbsTol must be positive.');  % 详解: 调用函数：error('AbsTol must be positive.')
end  % 详解: 执行语句

normcontrol = strcmp(odeget(options,'NormControl','off'),'on');  % 详解: 赋值：将 strcmp(...) 的结果保存到 normcontrol
if normcontrol  % 详解: 条件判断：if (normcontrol)
  if length(atol) ~= 1  % 详解: 条件判断：if (length(atol) ~= 1)
    error('Solving with NormControl ''on'' requires a scalar AbsTol.');  % 详解: 调用函数：error('Solving with NormControl ''on'' requires a scalar AbsTol.')
  end  % 详解: 执行语句
  normy = norm(y);  % 详解: 赋值：将 norm(...) 的结果保存到 normy
else  % 详解: 条件判断：else 分支
  if (length(atol) ~= 1) & (length(atol) ~= neq)  % 详解: 条件判断：if ((length(atol) ~= 1) & (length(atol) ~= neq))
    error(sprintf(['Solving %s requires a scalar AbsTol, ' ...  % 详解: 执行语句
                   'or a vector AbsTol of length %d'],upper(odefile),neq));  % 详解: 执行语句
  end  % 详解: 执行语句
  atol = atol(:);  % 详解: 赋值：将 atol(...) 的结果保存到 atol
end  % 详解: 执行语句
threshold = atol / rtol;  % 详解: 赋值：计算表达式并保存到 threshold

hmax = min(abs(tfinal-t), abs(odeget(options,'MaxStep',0.1*(tfinal-t))));  % 详解: 赋值：将 min(...) 的结果保存到 hmax
if hmax <= 0  % 详解: 条件判断：if (hmax <= 0)
  error('Option ''MaxStep'' must be greater than zero.');  % 详解: 调用函数：error('Option ''MaxStep'' must be greater than zero.')
end  % 详解: 执行语句
htry = abs(odeget(options,'InitialStep'));  % 详解: 赋值：将 abs(...) 的结果保存到 htry
if htry <= 0  % 详解: 条件判断：if (htry <= 0)
  error('Option ''InitialStep'' must be greater than zero.');  % 详解: 调用函数：error('Option ''InitialStep'' must be greater than zero.')
end  % 详解: 执行语句

haveeventfun = strcmp(odeget(options,'Events','off'),'on');  % 详解: 赋值：将 strcmp(...) 的结果保存到 haveeventfun
if haveeventfun  % 详解: 条件判断：if (haveeventfun)
  valt = feval(odefile,t,y,'events',varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 valt
  teout = [];  % 详解: 赋值：计算表达式并保存到 teout
  yeout = [];  % 详解: 赋值：计算表达式并保存到 yeout
  ieout = [];  % 详解: 赋值：计算表达式并保存到 ieout
end  % 详解: 执行语句

if nargout > 0  % 详解: 条件判断：if (nargout > 0)
  outfun = odeget(options,'OutputFcn');  % 详解: 赋值：将 odeget(...) 的结果保存到 outfun
else  % 详解: 条件判断：else 分支
  outfun = odeget(options,'OutputFcn','odeplot');  % 详解: 赋值：将 odeget(...) 的结果保存到 outfun
end  % 详解: 执行语句
if isempty(outfun)  % 详解: 条件判断：if (isempty(outfun))
  haveoutfun = false;  % 详解: 赋值：计算表达式并保存到 haveoutfun
else  % 详解: 条件判断：else 分支
  haveoutfun = true;  % 详解: 赋值：计算表达式并保存到 haveoutfun
  outputs = odeget(options,'OutputSel',1:neq);  % 详解: 赋值：将 odeget(...) 的结果保存到 outputs
end  % 详解: 执行语句
refine = odeget(options,'Refine',4);  % 详解: 赋值：将 odeget(...) 的结果保存到 refine
printstats = strcmp(odeget(options,'Stats','off'),'on');  % 详解: 赋值：将 strcmp(...) 的结果保存到 printstats

if strcmp(odeget(options,'Mass','off'),'on') | ...  % 详解: 条件判断：if (strcmp(odeget(options,'Mass','off'),'on') | ...)
  strcmp(odeget(options,'MassConstant','off'),'on')  % 详解: 调用函数：strcmp(odeget(options,'MassConstant','off'),'on')
  error(['Solver does not handle mass matrices, M*y'' or M(t)*y''.  '...  % 详解: 执行语句
         'See ODE15S, ODE23S, ODE23T, or ODE23TB.']);  % 详解: 执行语句
end  % 详解: 执行语句

if ntspan > 2  % 详解: 条件判断：if (ntspan > 2)
  outflag = 1;  % 详解: 赋值：计算表达式并保存到 outflag
elseif refine <= 1  % 详解: 条件判断：elseif (refine <= 1)
  outflag = 2;  % 详解: 赋值：计算表达式并保存到 outflag
else  % 详解: 条件判断：else 分支
  outflag = 3;  % 详解: 赋值：计算表达式并保存到 outflag
  S = (1:refine-1)' / refine;  % 赋值：设置变量 S  % 详解: 赋值：计算表达式并保存到 S  % 详解: 赋值：计算表达式并保存到 S
end  % 详解: 执行语句

if nargout > 0  % 详解: 条件判断：if (nargout > 0)
  if ntspan > 2  % 详解: 条件判断：if (ntspan > 2)
    tout = zeros(ntspan,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 tout
    yout = zeros(ntspan,neq);  % 详解: 赋值：将 zeros(...) 的结果保存到 yout
  else  % 详解: 条件判断：else 分支
    chunk = max(ceil(128 / neq),refine);  % 详解: 赋值：将 max(...) 的结果保存到 chunk
    tout = zeros(chunk,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 tout
    yout = zeros(chunk,neq);  % 详解: 赋值：将 zeros(...) 的结果保存到 yout
  end  % 详解: 执行语句
  nout = 1;  % 详解: 赋值：计算表达式并保存到 nout
  tout(nout) = t;  % 详解: 执行语句
  yout(nout,:) = y.';  % 调用函数：yout  % 详解: 执行语句  % 详解: 执行语句
end  % 详解: 执行语句

pow = 1/5;  % 详解: 赋值：计算表达式并保存到 pow
A = [1/5; 3/10; 4/5; 8/9; 1; 1];  % 详解: 赋值：计算表达式并保存到 A
B = [  % 详解: 赋值：计算表达式并保存到 B
    1/5         3/40    44/45   19372/6561      9017/3168       35/384  % 详解: 执行语句
    0           9/40    -56/15  -25360/2187     -355/33         0  % 详解: 执行语句
    0           0       32/9    64448/6561      46732/5247      500/1113  % 详解: 执行语句
    0           0       0       -212/729        49/176          125/192  % 详解: 执行语句
    0           0       0       0               -5103/18656     -2187/6784  % 详解: 执行语句
    0           0       0       0               0               11/84  % 详解: 执行语句
    0           0       0       0               0               0  % 详解: 执行语句
    ];  % 详解: 执行语句
E = [71/57600; 0; -71/16695; 71/1920; -17253/339200; 22/525; -1/40];  % 详解: 赋值：计算表达式并保存到 E
f = zeros(neq,7);  % 详解: 赋值：将 zeros(...) 的结果保存到 f

if nargin(odefile) == 2  % 详解: 条件判断：if (nargin(odefile) == 2)
  args = {};  % 详解: 赋值：计算表达式并保存到 args
else  % 详解: 条件判断：else 分支
  args = [{''} varargin];  % 详解: 赋值：计算表达式并保存到 args
end  % 详解: 执行语句

f0 = feval(odefile,t,y,args{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f0
nfevals = nfevals + 1;  % 详解: 赋值：计算表达式并保存到 nfevals
[m,n] = size(f0);  % 详解: 获取向量/矩阵尺寸
if n > 1  % 详解: 条件判断：if (n > 1)
  error([upper(odefile) ' must return a column vector.'])  % 详解: 调用函数：error([upper(odefile) ' must return a column vector.'])
elseif m ~= neq  % 详解: 条件判断：elseif (m ~= neq)
  msg = sprintf('an initial condition vector of length %d.',m);  % 详解: 赋值：将 sprintf(...) 的结果保存到 msg
  error(['Solving ' upper(odefile) ' requires ' msg]);  % 详解: 调用函数：error(['Solving ' upper(odefile) ' requires ' msg])
end  % 详解: 执行语句

hmin = 16*eps*abs(t);  % 详解: 赋值：计算表达式并保存到 hmin
if isempty(htry)  % 详解: 条件判断：if (isempty(htry))
  absh = min(hmax, abs(tspan(next) - t));  % 详解: 赋值：将 min(...) 的结果保存到 absh
  if normcontrol  % 详解: 条件判断：if (normcontrol)
    rh = (norm(f0) / max(normy,threshold)) / (0.8 * rtol^pow);  % 详解: 赋值：计算表达式并保存到 rh
  else  % 详解: 条件判断：else 分支
    rh = norm(f0 ./ max(abs(y),threshold),inf) / (0.8 * rtol^pow);  % 详解: 赋值：将 norm(...) 的结果保存到 rh
  end  % 详解: 执行语句
  if absh * rh > 1  % 详解: 条件判断：if (absh * rh > 1)
    absh = 1 / rh;  % 详解: 赋值：计算表达式并保存到 absh
  end  % 详解: 执行语句
  absh = max(absh, hmin);  % 详解: 赋值：将 max(...) 的结果保存到 absh
else  % 详解: 条件判断：else 分支
  absh = min(hmax, max(hmin, htry));  % 详解: 赋值：将 min(...) 的结果保存到 absh
end  % 详解: 执行语句
f(:,1) = f0;  % 详解: 执行语句

if haveoutfun  % 详解: 条件判断：if (haveoutfun)
  feval(outfun,[t tfinal],y(outputs),'init');  % 详解: 调用函数：feval(outfun,[t tfinal],y(outputs),'init')
end  % 详解: 执行语句


done = false;  % 详解: 赋值：计算表达式并保存到 done
while ~done  % 详解: while 循环：当 (~done) 为真时迭代
  
  hmin = 16*eps*abs(t);  % 详解: 赋值：计算表达式并保存到 hmin
  absh = min(hmax, max(hmin, absh));  % 详解: 赋值：将 min(...) 的结果保存到 absh
  h = tdir * absh;  % 详解: 赋值：计算表达式并保存到 h
  
  if 1.1*absh >= abs(tfinal - t)  % 详解: 条件判断：if (1.1*absh >= abs(tfinal - t))
    h = tfinal - t;  % 详解: 赋值：计算表达式并保存到 h
    absh = abs(h);  % 详解: 赋值：将 abs(...) 的结果保存到 absh
    done = true;  % 详解: 赋值：计算表达式并保存到 done
  end  % 详解: 执行语句
  
  nofailed = true;  % 详解: 赋值：计算表达式并保存到 nofailed
  while true  % 详解: while 循环：当 (true) 为真时迭代
    hA = h * A;  % 详解: 赋值：计算表达式并保存到 hA
    hB = h * B;  % 详解: 赋值：计算表达式并保存到 hB
    f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args{:});  % 详解: 调用函数：f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args{:})
    f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args{:});  % 详解: 调用函数：f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args{:})
    f(:,4) = feval(odefile, t + hA(3), y + f*hB(:,3), args{:});  % 详解: 调用函数：f(:,4) = feval(odefile, t + hA(3), y + f*hB(:,3), args{:})
    f(:,5) = feval(odefile, t + hA(4), y + f*hB(:,4), args{:});  % 详解: 调用函数：f(:,5) = feval(odefile, t + hA(4), y + f*hB(:,4), args{:})
    f(:,6) = feval(odefile, t + hA(5), y + f*hB(:,5), args{:});  % 详解: 调用函数：f(:,6) = feval(odefile, t + hA(5), y + f*hB(:,5), args{:})
    tnew = t + hA(6);  % 详解: 赋值：计算表达式并保存到 tnew
    ynew = y + f*hB(:,6);  % 详解: 赋值：计算表达式并保存到 ynew
    f(:,7) = feval(odefile, tnew, ynew, args{:});  % 详解: 调用函数：f(:,7) = feval(odefile, tnew, ynew, args{:})
    nfevals = nfevals + 6;  % 详解: 赋值：计算表达式并保存到 nfevals
    
    if normcontrol  % 详解: 条件判断：if (normcontrol)
      normynew = norm(ynew);  % 详解: 赋值：将 norm(...) 的结果保存到 normynew
      err = absh * (norm(f * E) / max(max(normy,normynew),threshold));  % 详解: 赋值：计算表达式并保存到 err
    else  % 详解: 条件判断：else 分支
      err = absh * norm((f * E) ./ max(max(abs(y),abs(ynew)),threshold),inf);  % 详解: 赋值：计算表达式并保存到 err
    end  % 详解: 执行语句
    
    if err > rtol  % 详解: 条件判断：if (err > rtol)
      nfailed = nfailed + 1;  % 详解: 赋值：计算表达式并保存到 nfailed
      if absh <= hmin  % 详解: 条件判断：if (absh <= hmin)
        msg = sprintf(['Failure at t=%e.  Unable to meet integration ' ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 msg
                       'tolerances without reducing the step size below ' ...  % 详解: 执行语句
                       'the smallest value allowed (%e) at time t.\n'],t,hmin);  % 详解: 执行语句
        warning(msg);  % 详解: 调用函数：warning(msg)
        if haveoutfun  % 详解: 条件判断：if (haveoutfun)
          feval(outfun,[],[],'done');  % 详解: 调用函数：feval(outfun,[],[],'done')
        end  % 详解: 执行语句
        if printstats  % 详解: 条件判断：if (printstats)
          fprintf('%g successful steps\n', nsteps);  % 详解: 调用函数：fprintf('%g successful steps\n', nsteps)
          fprintf('%g failed attempts\n', nfailed);  % 详解: 调用函数：fprintf('%g failed attempts\n', nfailed)
          fprintf('%g function evaluations\n', nfevals);  % 详解: 调用函数：fprintf('%g function evaluations\n', nfevals)
          fprintf('%g partial derivatives\n', npds);  % 详解: 调用函数：fprintf('%g partial derivatives\n', npds)
          fprintf('%g LU decompositions\n', ndecomps);  % 详解: 调用函数：fprintf('%g LU decompositions\n', ndecomps)
          fprintf('%g solutions of linear systems\n', nsolves);  % 详解: 调用函数：fprintf('%g solutions of linear systems\n', nsolves)
        end  % 详解: 执行语句
        if nargout > 0  % 详解: 条件判断：if (nargout > 0)
          tout = tout(1:nout);  % 详解: 赋值：将 tout(...) 的结果保存到 tout
          yout = yout(1:nout,:);  % 详解: 赋值：将 yout(...) 的结果保存到 yout
          if haveeventfun  % 详解: 条件判断：if (haveeventfun)
            varargout{1} = teout;  % 详解: 执行语句
            varargout{2} = yeout;  % 详解: 执行语句
            varargout{3} = ieout;  % 详解: 执行语句
            varargout{4} = [nsteps; nfailed; nfevals; npds; ndecomps; nsolves];  % 详解: 执行语句
          else  % 详解: 条件判断：else 分支
            varargout{1} = [nsteps; nfailed; nfevals; npds; ndecomps; nsolves];  % 详解: 执行语句
          end  % 详解: 执行语句
        end  % 详解: 执行语句
        return;  % 详解: 返回：从当前函数返回
      end  % 详解: 执行语句
      
      if nofailed  % 详解: 条件判断：if (nofailed)
        nofailed = false;  % 详解: 赋值：计算表达式并保存到 nofailed
        absh = max(hmin, absh * max(0.1, 0.8*(rtol/err)^pow));  % 详解: 赋值：将 max(...) 的结果保存到 absh
      else  % 详解: 条件判断：else 分支
        absh = max(hmin, 0.5 * absh);  % 详解: 赋值：将 max(...) 的结果保存到 absh
      end  % 详解: 执行语句
      h = tdir * absh;  % 详解: 赋值：计算表达式并保存到 h
      done = false;  % 详解: 赋值：计算表达式并保存到 done
      
    else  % 详解: 条件判断：else 分支
      break;  % 详解: 跳出循环：break
      
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  nsteps = nsteps + 1;  % 详解: 赋值：计算表达式并保存到 nsteps
  
  if haveeventfun  % 详解: 条件判断：if (haveeventfun)
    [te,ye,ie,valt,stop] = ...  % 详解: 执行语句
        odezero('ntrp45',odefile,valt,t,y,tnew,ynew,t0,varargin,h,f);  % 详解: 调用函数：odezero('ntrp45',odefile,valt,t,y,tnew,ynew,t0,varargin,h,f)
    nte = length(te);  % 详解: 赋值：将 length(...) 的结果保存到 nte
    if nte > 0  % 详解: 条件判断：if (nte > 0)
      if nargout > 2  % 详解: 条件判断：if (nargout > 2)
        teout = [teout; te];  % 详解: 赋值：计算表达式并保存到 teout
        yeout = [yeout; ye.'];  % 赋值：设置变量 yeout  % 详解: 赋值：计算表达式并保存到 yeout  % 详解: 赋值：计算表达式并保存到 yeout
        ieout = [ieout; ie];  % 详解: 赋值：计算表达式并保存到 ieout
      end  % 详解: 执行语句
      if stop  % 详解: 条件判断：if (stop)
        tnew = te(nte);  % 详解: 赋值：将 te(...) 的结果保存到 tnew
        ynew = ye(:,nte);  % 详解: 赋值：将 ye(...) 的结果保存到 ynew
        done = true;  % 详解: 赋值：计算表达式并保存到 done
      end  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  
  if nargout > 0  % 详解: 条件判断：if (nargout > 0)
    oldnout = nout;  % 详解: 赋值：计算表达式并保存到 oldnout
    if outflag == 3  % 详解: 条件判断：if (outflag == 3)
      nout = nout + refine;  % 详解: 赋值：计算表达式并保存到 nout
      if nout > length(tout)  % 详解: 条件判断：if (nout > length(tout))
        tout = [tout; zeros(chunk,1)];  % 详解: 赋值：计算表达式并保存到 tout
        yout = [yout; zeros(chunk,neq)];  % 详解: 赋值：计算表达式并保存到 yout
      end  % 详解: 执行语句
      i = oldnout+1:nout-1;  % 详解: 赋值：计算表达式并保存到 i
      tout(i) = t + (tnew-t)*S;  % 详解: 执行语句
      yout(i,:) = ntrp45(tout(i),t,y,[],[],h,f).';  % 调用函数：yout  % 详解: 执行语句  % 详解: 执行语句
      tout(nout) = tnew;  % 详解: 执行语句
      yout(nout,:) = ynew.';  % 调用函数：yout  % 详解: 执行语句  % 详解: 执行语句
    elseif outflag == 2  % 详解: 条件判断：elseif (outflag == 2)
      nout = nout + 1;  % 详解: 赋值：计算表达式并保存到 nout
      if nout > length(tout)  % 详解: 条件判断：if (nout > length(tout))
        tout = [tout; zeros(chunk,1)];  % 详解: 赋值：计算表达式并保存到 tout
        yout = [yout; zeros(chunk,neq)];  % 详解: 赋值：计算表达式并保存到 yout
      end  % 详解: 执行语句
      tout(nout) = tnew;  % 详解: 执行语句
      yout(nout,:) = ynew.';  % 调用函数：yout  % 详解: 执行语句  % 详解: 执行语句
    elseif outflag == 1  % 详解: 条件判断：elseif (outflag == 1)
      while next <= ntspan  % 详解: while 循环：当 (next <= ntspan) 为真时迭代
        if tdir * (tnew - tspan(next)) < 0  % 详解: 条件判断：if (tdir * (tnew - tspan(next)) < 0)
          if haveeventfun & done  % 详解: 条件判断：if (haveeventfun & done)
            nout = nout + 1;  % 详解: 赋值：计算表达式并保存到 nout
            tout(nout) = tnew;  % 详解: 执行语句
            yout(nout,:) = ynew.';  % 调用函数：yout  % 详解: 执行语句  % 详解: 执行语句
          end  % 详解: 执行语句
          break;  % 详解: 跳出循环：break
        elseif tnew == tspan(next)  % 详解: 条件判断：elseif (tnew == tspan(next))
          nout = nout + 1;  % 详解: 赋值：计算表达式并保存到 nout
          tout(nout) = tnew;  % 详解: 执行语句
          yout(nout,:) = ynew.';  % 调用函数：yout  % 详解: 执行语句  % 详解: 执行语句
          next = next + 1;  % 详解: 赋值：计算表达式并保存到 next
          break;  % 详解: 跳出循环：break
        end  % 详解: 执行语句
        nout = nout + 1;  % 详解: 赋值：计算表达式并保存到 nout
        tout(nout) = tspan(next);  % 详解: 调用函数：tout(nout) = tspan(next)
        yout(nout,:) = ntrp45(tspan(next),t,y,[],[],h,f).';  % 调用函数：yout  % 详解: 执行语句  % 详解: 执行语句
        next = next + 1;  % 详解: 赋值：计算表达式并保存到 next
      end  % 详解: 执行语句
    end  % 详解: 执行语句
    
    if haveoutfun  % 详解: 条件判断：if (haveoutfun)
      i = oldnout+1:nout;  % 详解: 赋值：计算表达式并保存到 i
      if ~isempty(i) & (feval(outfun,tout(i),yout(i,outputs).') == 1)  % 条件判断：if 分支开始  % 详解: 条件判断：if (~isempty(i) & (feval(outfun,tout(i),yout(i,outputs).') == 1))  % 详解: 条件判断：if (~isempty(i) & (feval(outfun,tout(i),yout(i,outputs).') == 1)  % 条件判断：if 分支开始  % 详解: 条件判断：if (~isempty(i) & (feval(outfun,tout(i),yout(i,outputs).') == 1)))
        tout = tout(1:nout);  % 详解: 赋值：将 tout(...) 的结果保存到 tout
        yout = yout(1:nout,:);  % 详解: 赋值：将 yout(...) 的结果保存到 yout
        if haveeventfun  % 详解: 条件判断：if (haveeventfun)
          varargout{1} = teout;  % 详解: 执行语句
          varargout{2} = yeout;  % 详解: 执行语句
          varargout{3} = ieout;  % 详解: 执行语句
          varargout{4} = [nsteps; nfailed; nfevals; npds; ndecomps; nsolves];  % 详解: 执行语句
        else  % 详解: 条件判断：else 分支
          varargout{1} = [nsteps; nfailed; nfevals; npds; ndecomps; nsolves];  % 详解: 执行语句
        end  % 详解: 执行语句
        return;  % 详解: 返回：从当前函数返回
      end  % 详解: 执行语句
    end  % 详解: 执行语句
    
  elseif haveoutfun  % 详解: 条件判断：elseif (haveoutfun)
    if outflag == 3  % 详解: 条件判断：if (outflag == 3)
      tinterp = t + (tnew-t)*S;  % 详解: 赋值：计算表达式并保存到 tinterp
      yinterp = ntrp45(tinterp,t,y,[],[],h,f);  % 详解: 赋值：将 ntrp45(...) 的结果保存到 yinterp
      if feval(outfun,[tinterp; tnew],[yinterp(outputs,:), ynew(outputs)]) == 1  % 详解: 条件判断：if (feval(outfun,[tinterp; tnew],[yinterp(outputs,:), ynew(outputs)]) == 1)
        return;  % 详解: 返回：从当前函数返回
      end  % 详解: 执行语句
    elseif outflag == 2  % 详解: 条件判断：elseif (outflag == 2)
      if feval(outfun,tnew,ynew(outputs)) == 1  % 详解: 条件判断：if (feval(outfun,tnew,ynew(outputs)) == 1)
        return;  % 详解: 返回：从当前函数返回
      end  % 详解: 执行语句
    elseif outflag == 1  % 详解: 条件判断：elseif (outflag == 1)
      ninterp = 0;  % 详解: 赋值：计算表达式并保存到 ninterp
      while next <= ntspan  % 详解: while 循环：当 (next <= ntspan) 为真时迭代
        if tdir * (tnew - tspan(next)) < 0  % 详解: 条件判断：if (tdir * (tnew - tspan(next)) < 0)
          if haveeventfun & done  % 详解: 条件判断：if (haveeventfun & done)
            ninterp = ninterp + 1;  % 详解: 赋值：计算表达式并保存到 ninterp
            tinterp(ninterp,1) = tnew;  % 详解: 执行语句
            yinterp(:,ninterp) = ynew;  % 详解: 执行语句
          end  % 详解: 执行语句
          break;  % 详解: 跳出循环：break
        elseif tnew == tspan(next)  % 详解: 条件判断：elseif (tnew == tspan(next))
          ninterp = ninterp + 1;  % 详解: 赋值：计算表达式并保存到 ninterp
          tinterp(ninterp,1) = tnew;  % 详解: 执行语句
          yinterp(:,ninterp) = ynew;  % 详解: 执行语句
          next = next + 1;  % 详解: 赋值：计算表达式并保存到 next
          break;  % 详解: 跳出循环：break
        end  % 详解: 执行语句
        ninterp = ninterp + 1;  % 详解: 赋值：计算表达式并保存到 ninterp
        tinterp(ninterp,1) = tspan(next);  % 详解: 调用函数：tinterp(ninterp,1) = tspan(next)
        yinterp(:,ninterp) = ntrp45(tspan(next),t,y,[],[],h,f);  % 详解: 调用函数：yinterp(:,ninterp) = ntrp45(tspan(next),t,y,[],[],h,f)
        next = next + 1;  % 详解: 赋值：计算表达式并保存到 next
      end  % 详解: 执行语句
      if ninterp > 0  % 详解: 条件判断：if (ninterp > 0)
        if feval(outfun,tinterp(1:ninterp),yinterp(outputs,1:ninterp)) == 1  % 详解: 条件判断：if (feval(outfun,tinterp(1:ninterp),yinterp(outputs,1:ninterp)) == 1)
          return;  % 详解: 返回：从当前函数返回
        end  % 详解: 执行语句
      end  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  
  if nofailed  % 详解: 条件判断：if (nofailed)
    temp = 1.25*(err/rtol)^pow;  % 详解: 赋值：计算表达式并保存到 temp
    if temp > 0.2  % 详解: 条件判断：if (temp > 0.2)
      absh = absh / temp;  % 详解: 赋值：计算表达式并保存到 absh
    else  % 详解: 条件判断：else 分支
      absh = 5.0*absh;  % 详解: 赋值：计算表达式并保存到 absh
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  
  t = tnew;  % 详解: 赋值：计算表达式并保存到 t
  y = ynew;  % 详解: 赋值：计算表达式并保存到 y
  if normcontrol  % 详解: 条件判断：if (normcontrol)
    normy = normynew;  % 详解: 赋值：计算表达式并保存到 normy
  end  % 详解: 执行语句
  f(:,1) = f(:,7);  % 详解: 调用函数：f(:,1) = f(:,7)
  
end  % 详解: 执行语句

if haveoutfun  % 详解: 条件判断：if (haveoutfun)
  feval(outfun,[],[],'done');  % 详解: 调用函数：feval(outfun,[],[],'done')
end  % 详解: 执行语句

if printstats  % 详解: 条件判断：if (printstats)
  fprintf('%g successful steps\n', nsteps);  % 详解: 调用函数：fprintf('%g successful steps\n', nsteps)
  fprintf('%g failed attempts\n', nfailed);  % 详解: 调用函数：fprintf('%g failed attempts\n', nfailed)
  fprintf('%g function evaluations\n', nfevals);  % 详解: 调用函数：fprintf('%g function evaluations\n', nfevals)
  fprintf('%g partial derivatives\n', npds);  % 详解: 调用函数：fprintf('%g partial derivatives\n', npds)
  fprintf('%g LU decompositions\n', ndecomps);  % 详解: 调用函数：fprintf('%g LU decompositions\n', ndecomps)
  fprintf('%g solutions of linear systems\n', nsolves);  % 详解: 调用函数：fprintf('%g solutions of linear systems\n', nsolves)
end  % 详解: 执行语句

if nargout > 0  % 详解: 条件判断：if (nargout > 0)
  tout = tout(1:nout);  % 详解: 赋值：将 tout(...) 的结果保存到 tout
  yout = yout(1:nout,:);  % 详解: 赋值：将 yout(...) 的结果保存到 yout
  if haveeventfun  % 详解: 条件判断：if (haveeventfun)
    varargout{1} = teout;  % 详解: 执行语句
    varargout{2} = yeout;  % 详解: 执行语句
    varargout{3} = ieout;  % 详解: 执行语句
    varargout{4} = [nsteps; nfailed; nfevals; npds; ndecomps; nsolves];  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    varargout{1} = [nsteps; nfailed; nfevals; npds; ndecomps; nsolves];  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句




