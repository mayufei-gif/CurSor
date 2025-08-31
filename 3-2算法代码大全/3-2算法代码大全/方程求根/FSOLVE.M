% 文件: FSOLVE.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加


% FSOLVE：采用最小二乘法求解非线性方程（组）
%
% 基本形式
%   求解 F(X) = 0，其中 F 与 X 可以是向量或矩阵。
%
% 基本用法
%   X = FSOLVE(FUN, X0)
%     从初始点/矩阵 X0 出发，尝试求解由 FUN 描述的方程。
%     FUN 通常是一个 M 文件（或函数句柄），对给定 X 返回方程的函数值：F = FUN(X)。
%
%   X = FSOLVE(FUN, X0, OPTIONS)
%     使用由 OPTIMSET 生成的 OPTIONS 结构体中的参数值替换默认优化参数。
%     相关选项包括：Display, TolX, TolFun, DerivativeCheck, Diagnostics, Jacobian,
%     JacobPattern, LineSearchType, LevenbergMarquardt, MaxFunEvals, MaxIter,
%     DiffMinChange, DiffMaxChange, LargeScale, MaxPCGIter, PrecondBandWidth,
%     TolPCG, TypicalX。
%     若设置 Jacobian 选项，FUN 可能会以两个输出被调用：第二个输出 J 为雅可比矩阵：
%       [F, J] = FUN(X)
%     若 FUN 返回长度为 m 的向量（或 m×1 矩阵），且 X 长度为 n，则 J 为 m×n 矩阵，
%     其中 J(i,j) 为 F(i) 关于 x(j) 的偏导数。（注意：雅可比矩阵 J 是 F 的梯度的转置。）
%
%   X = FSOLVE(FUN, X0, OPTIONS, P1, P2, ...)
%     将问题相关的参数 P1, P2, ... 直接传入 FUN：FUN(X, P1, P2, …)。
%     若要使用默认选项，请传入空矩阵 [] 作为 OPTIONS。
%
% 返回值说明
%   [X, FVAL] = FSOLVE(FUN, X0, ...)
%     返回在解 X 处的目标函数值 FVAL（即方程残差的度量）。
%
%   [X, FVAL, EXITFLAG] = FSOLVE(FUN, X0, ...)
%     返回字符串/标志 EXITFLAG，描述 FSOLVE 的退出情况：
%       > 0 表示已收敛到解 X；
%         0 表示达到最大函数评估次数；
%       < 0 表示未能收敛到解。
%
%   [X, FVAL, EXITFLAG, OUTPUT] = FSOLVE(FUN, X0, ...)
%     返回结构体 OUTPUT：包含迭代次数 OUTPUT.iterations、函数评估次数 OUTPUT.funcCount、
%     使用的算法 OUTPUT.algorithm、（如使用）共轭梯度迭代次数 OUTPUT.cgiterations、
%     以及一阶最优性指标 OUTPUT.firstorderopt（如使用）。
%
%   [X, FVAL, EXITFLAG, OUTPUT, JACOB] = FSOLVE(FUN, X0, ...)
%     额外返回在解 X 处的雅可比矩阵 JACOB。
%
% 版权信息
%   Copyright (c) 1990–1998 The MathWorks, Inc.
%   $Revision: 1.26 $  $Date: 1998/10/22 19:28:31 $
%   Andy Grace 7-9-90.
%
% 兼容性说明（Optimization Toolbox 2.0 之前）
%   旧式调用：
%     [X, OPTIONS] = FSOLVE(FUN, X0, OPTIONS, GRADFUN, P1, P2, ...)
%
% ------------初始化（Initialization）----------------

function [x,FVAL,EXITFLAG,OUTPUT,JACOB] = fsolve(FUN,x,options,varargin)
defaultopt = optimset('display','final','LargeScale','on', ...
   'TolX',1e-6,'TolFun',1e-6,'DerivativeCheck','off',...
   'Jacobian','off','MaxFunEvals','100*numberOfVariables',...
   'Diagnostics','off',...
   'DiffMaxChange',1e-1,'DiffMinChange',1e-8,...
   'PrecondBandWidth',0,'TypicalX','ones(numberOfVariables,1)','MaxPCGIter','max(1,floor(numberOfVariables/2))', ...
   'TolPCG',0.1,'MaxIter',400,'JacobPattern',[], ...
   'LineSearchType','quadcubic','LevenbergMarq','off');
% If just 'defaults' passed in, return the default options in X
if nargin==1 && nargout <= 1 && isequal(FUN,'defaults')
   x = defaultopt;
   return
end

if nargin < 2, error('FSOLVE requires two input arguments');end  % 详解: 条件判断：if (nargin < 2, error('FSOLVE requires two input arguments');end)
if nargin < 3, options=[]; end  % 详解: 条件判断：if (nargin < 3, options=[]; end)

LB = []; UB = [];  % 详解: 赋值：计算表达式并保存到 LB

numargin = nargin; numargout = nargout;  % 详解: 赋值：计算表达式并保存到 numargin
[calltype, GRADFUN, varargin] = parse_call(FUN,options,numargin,numargout,varargin);  % 详解: 执行语句

if isequal(calltype,'new')  % 详解: 条件判断：if (isequal(calltype,'new'))
   
   xstart=x(:);  % 详解: 赋值：将 x(...) 的结果保存到 xstart
   numberOfVariables=length(xstart);  % 详解: 赋值：将 length(...) 的结果保存到 numberOfVariables
   
   large = 'large-scale';  % 详解: 赋值：计算表达式并保存到 large
   medium = 'medium-scale';  % 详解: 赋值：计算表达式并保存到 medium
   
   l = []; u = [];  % 详解: 赋值：计算表达式并保存到 l
   
   options = optimset(defaultopt,options);  % 详解: 赋值：将 optimset(...) 的结果保存到 options
   switch optimget(options,'display')  % 详解: 多分支选择：switch (optimget(options,'display'))
   case {'off','none'}  % 详解: 分支：case {'off','none'}
      verbosity = 0;  % 详解: 赋值：计算表达式并保存到 verbosity
   case 'iter'  % 详解: 分支：case 'iter'
      verbosity = 2;  % 详解: 赋值：计算表达式并保存到 verbosity
   case 'final'  % 详解: 分支：case 'final'
      verbosity = 1;  % 详解: 赋值：计算表达式并保存到 verbosity
   case 'testing'  % 详解: 分支：case 'testing'
      verbosity = Inf;  % 详解: 赋值：计算表达式并保存到 verbosity
   otherwise  % 详解: 默认分支：otherwise
      verbosity = 1;  % 详解: 赋值：计算表达式并保存到 verbosity
   end  % 详解: 执行语句
   diagnostics = isequal(optimget(options,'diagnostics','off'),'on');  % 详解: 赋值：将 isequal(...) 的结果保存到 diagnostics
   
   gradflag =  strcmp(optimget(options,'Jacobian'),'on');  % 详解: 赋值：将 strcmp(...) 的结果保存到 gradflag
   line_search = strcmp(optimget(options,'largescale','off'),'off');  % 详解: 赋值：将 strcmp(...) 的结果保存到 line_search
   
   if ~isempty(FUN)  % 详解: 条件判断：if (~isempty(FUN))
      [funfcn, msg] = fprefcnchk(FUN,'fsolve',length(varargin),gradflag);  % 详解: 获取向量/矩阵尺寸
   else  % 详解: 条件判断：else 分支
      errmsg = sprintf('%s\n%s', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FUN must be a function name, valid string expression, or inline object;', ...  % 详解: 执行语句
         ' or, FUN may be a cell array that contains these type of objects.');  % 详解: 执行语句
      error(errmsg)  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
   
   x(:) = xstart;  % 详解: 执行语句
   switch funfcn{1}  % 详解: 多分支选择：switch (funfcn{1})
   case 'fun'  % 详解: 分支：case 'fun'
      fuser = feval(funfcn{3},x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 fuser
      f = fuser(:);  % 详解: 赋值：将 fuser(...) 的结果保存到 f
      nfun=length(f);  % 详解: 赋值：将 length(...) 的结果保存到 nfun
      JAC = zeros(nfun,numberOfVariables);  % 详解: 赋值：将 zeros(...) 的结果保存到 JAC
   case 'fungrad'  % 详解: 分支：case 'fungrad'
      [fuser,JAC] = feval(funfcn{3},x,varargin{:});  % 详解: 执行语句
      f = fuser(:);  % 详解: 赋值：将 fuser(...) 的结果保存到 f
      nfun=length(f);  % 详解: 赋值：将 length(...) 的结果保存到 nfun
   case 'fun_then_grad'  % 详解: 分支：case 'fun_then_grad'
      fuser = feval(funfcn{3},x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 fuser
      f = fuser(:);  % 详解: 赋值：将 fuser(...) 的结果保存到 f
      JAC = feval(funfcn{4},x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 JAC
      nfun=length(f);  % 详解: 赋值：将 length(...) 的结果保存到 nfun
      
   otherwise  % 详解: 默认分支：otherwise
      error('Undefined calltype in FSOLVE');  % 详解: 调用函数：error('Undefined calltype in FSOLVE')
   end  % 详解: 执行语句
   
   [Jrows, Jcols]=size(JAC);  % 详解: 获取向量/矩阵尺寸
   if Jrows~=nfun | Jcols ~=numberOfVariables  % 详解: 条件判断：if (Jrows~=nfun | Jcols ~=numberOfVariables)
      errstr = sprintf('%s\n%s%d%s%d\n',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errstr
         'User-defined Jacobian is not the correct size:',...  % 详解: 执行语句
         '    the Jacobian matrix should be ',nfun,'-by-',numberOfVariables);  % 详解: 执行语句
      error(errstr);  % 详解: 调用函数：error(errstr)
   end  % 详解: 执行语句
   
   YDATA = []; caller = 'fsolve';  % 详解: 赋值：计算表达式并保存到 YDATA
   
   if ~line_search & nfun >= numberOfVariables  % 详解: 条件判断：if (~line_search & nfun >= numberOfVariables)
      OUTPUT.algorithm = large;  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
      
   elseif ~line_search & nfun < numberOfVariables  % 详解: 条件判断：elseif (~line_search & nfun < numberOfVariables)
      warnstr = sprintf('%s\n%s\n', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 warnstr
         'Large-scale method requires at least as many equations as variables; ',...  % 详解: 执行语句
         '   switching to line-search method instead.');  % 详解: 执行语句
      warning(warnstr);  % 详解: 调用函数：warning(warnstr)
      OUTPUT.algorithm = medium;  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
      
   elseif line_search & isempty(l) & isempty(u)  % 详解: 条件判断：elseif (line_search & isempty(l) & isempty(u))
      OUTPUT.algorithm = medium;  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
      
   elseif line_search & (~isempty(LB) | ~isempty(UB))  & nfun >= numberOfVariables  % 详解: 条件判断：elseif (line_search & (~isempty(LB) | ~isempty(UB))  & nfun >= numberOfVariables)
      warnstr = sprintf('%s\n%s\n', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 warnstr
         'Line-search method does not handle bound constraints; ',...  % 详解: 执行语句
         '   switching to trust-region method instead.');  % 详解: 执行语句
      warning(warnstr);  % 详解: 调用函数：warning(warnstr)
      OUTPUT.algorithm = large;  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
      
   elseif line_search & (~isempty(LB) | ~isempty(UB))  & nfun < numberOfVariables  % 详解: 条件判断：elseif (line_search & (~isempty(LB) | ~isempty(UB))  & nfun < numberOfVariables)
      errstr = sprintf('%s\n%s\n%s\n', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errstr
         'Line-search method does not handle bound constraints ',...  % 详解: 执行语句
         '   and trust-region method requires at least as many equations as variables; ',...  % 详解: 执行语句
         '   aborting.');  % 详解: 执行语句
      error(errstr);  % 详解: 调用函数：error(errstr)
      
   end  % 详解: 执行语句
   
   if diagnostics > 0  % 详解: 条件判断：if (diagnostics > 0)
      constflag = 0; gradconstflag = 0; non_eq=0;non_ineq=0;lin_eq=0;lin_ineq=0;  % 详解: 赋值：计算表达式并保存到 constflag
      confcn{1}=[];c=[];ceq=[];cGRAD=[];ceqGRAD=[];  % 详解: 执行语句
      hessflag = 0; HESS=[];  % 详解: 赋值：计算表达式并保存到 hessflag
      msg = diagnose('fsolve',OUTPUT,gradflag,hessflag,constflag,gradconstflag,...  % 详解: 赋值：将 diagnose(...) 的结果保存到 msg
         line_search,options,xstart,non_eq,...  % 详解: 执行语句
         non_ineq,lin_eq,lin_ineq,LB,UB,funfcn,confcn,f,JAC,HESS,c,ceq,cGRAD,ceqGRAD);  % 详解: 执行语句
      
   end  % 详解: 执行语句
   
   if isequal(OUTPUT.algorithm, large)  % 详解: 条件判断：if (isequal(OUTPUT.algorithm, large))
      if ~gradflag  % 详解: 条件判断：if (~gradflag)
         Jstr = optimget(options,'JacobPattern',[]);  % 详解: 赋值：将 optimget(...) 的结果保存到 Jstr
         if isempty(Jstr)  % 详解: 条件判断：if (isempty(Jstr))
            Jstr = sparse(ones(Jrows,Jcols));  % 详解: 赋值：将 sparse(...) 的结果保存到 Jstr
         end  % 详解: 执行语句
      else  % 详解: 条件判断：else 分支
         Jstr = [];  % 详解: 赋值：计算表达式并保存到 Jstr
      end  % 详解: 执行语句
      l = []; u = []; computeLambda = 0;  % 详解: 赋值：计算表达式并保存到 l
      [x,FVAL,LAMBDA,JACOB,EXITFLAG,OUTPUT]=...  % 详解: 执行语句
         snls(funfcn,x,l,u,verbosity,options,f,JAC,YDATA,caller,Jstr,computeLambda,varargin{:});  % 详解: 调用函数：snls(funfcn,x,l,u,verbosity,options,f,JAC,YDATA,caller,Jstr,computeLambda,varargin{:})
   else  % 详解: 条件判断：else 分支
      [x,FVAL,JACOB,EXITFLAG,OUTPUT] = ...  % 详解: 执行语句
         nlsq(funfcn,x,verbosity,options,f,JAC,YDATA,caller,varargin{:});  % 详解: 调用函数：nlsq(funfcn,x,verbosity,options,f,JAC,YDATA,caller,varargin{:})
      
   end  % 详解: 执行语句
   
   Resnorm = FVAL'*FVAL;  % assumes FVAL still a vector  % 详解: 赋值：计算表达式并保存到 Resnorm  % 详解: 赋值：计算表达式并保存到 Resnorm
   if Resnorm > 10*optimget(options,'TolFun',1e-4) & verbosity>0  % 详解: 条件判断：if (Resnorm > 10*optimget(options,'TolFun',1e-4) & verbosity>0)
      if verbosity > 0  % 详解: 条件判断：if (verbosity > 0)
         disp('Optimizer is stuck at a minimum that is not a root')  % 详解: 调用函数：disp('Optimizer is stuck at a minimum that is not a root')
         disp('Try again with a new starting guess')  % 详解: 调用函数：disp('Try again with a new starting guess')
      end  % 详解: 执行语句
      EXITFLAG = -1;  % 详解: 赋值：计算表达式并保存到 EXITFLAG
   end  % 详解: 执行语句
   
   FVAL = reshape(FVAL,size(fuser));  % 详解: 赋值：将 reshape(...) 的结果保存到 FVAL
   
else  % 详解: 条件判断：else 分支
   
   if length(options)<5;  % 详解: 条件判断：if (length(options)<5;)
      options(5)=0;  % 详解: 执行语句
   end  % 详解: 执行语句
   if options(5)==0; options(5)=1; else options(5)=0; end  % 详解: 条件判断：if (options(5)==0; options(5)=1; else options(5)=0; end)
   
   if ~isempty(FUN)  % 详解: 条件判断：if (~isempty(FUN))
      [funfcn, msg] = fcnchk(FUN,length(varargin));  % 详解: 获取向量/矩阵尺寸
      if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
         error(msg);  % 详解: 调用函数：error(msg)
      end  % 详解: 执行语句
   else  % 详解: 条件判断：else 分支
      error('FUN must be a function name or valid expression.')  % 详解: 调用函数：error('FUN must be a function name or valid expression.')
   end  % 详解: 执行语句
   
   if ~isempty(GRADFUN)  % 详解: 条件判断：if (~isempty(GRADFUN))
      [gradfcn, msg] = fcnchk(GRADFUN,length(varargin));  % 详解: 获取向量/矩阵尺寸
      if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
         error(msg);  % 详解: 调用函数：error(msg)
      end  % 详解: 执行语句
   else  % 详解: 条件判断：else 分支
      gradfcn = [];  % 详解: 赋值：计算表达式并保存到 gradfcn
   end  % 详解: 执行语句
   
   [x,options] = nlsqold(funfcn,x,options,gradfcn,varargin{:});  % 详解: 执行语句
   
   if options(8)>10*options(3) & options(1)>0  % 详解: 条件判断：if (options(8)>10*options(3) & options(1)>0)
      disp('Optimizer is stuck at a minimum that is not a root')  % 详解: 调用函数：disp('Optimizer is stuck at a minimum that is not a root')
      disp('Try again with a new starting guess')  % 详解: 调用函数：disp('Try again with a new starting guess')
   end  % 详解: 执行语句
   
   FVAL = options;  % 详解: 赋值：计算表达式并保存到 FVAL
   
   
end  % 详解: 执行语句



function [allfcns,msg] = fprefcnchk(funstr,caller,lenVarIn,gradflag)  % 详解: 函数定义：fprefcnchk(funstr,caller,lenVarIn,gradflag), 返回：allfcns,msg
msg='';  % 详解: 赋值：计算表达式并保存到 msg
allfcns = {};  % 详解: 赋值：计算表达式并保存到 allfcns
funfcn = [];  % 详解: 赋值：计算表达式并保存到 funfcn
gradfcn = [];  % 详解: 赋值：计算表达式并保存到 gradfcn

if gradflag  % 详解: 条件判断：if (gradflag)
   calltype = 'fungrad';  % 详解: 赋值：计算表达式并保存到 calltype
else  % 详解: 条件判断：else 分支
   calltype = 'fun';  % 详解: 赋值：计算表达式并保存到 calltype
end  % 详解: 执行语句

if isa(funstr, 'cell') & length(funstr)==1  % 详解: 条件判断：if (isa(funstr, 'cell') & length(funstr)==1)
   if gradflag  % 详解: 条件判断：if (gradflag)
      calltype = 'fungrad';0  % 详解: 赋值：计算表达式并保存到 calltype
   end  % 详解: 执行语句
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   
elseif isa(funstr, 'cell') & length(funstr)==2 & isempty(funstr{2})  % 详解: 条件判断：elseif (isa(funstr, 'cell') & length(funstr)==2 & isempty(funstr{2}))
   if gradflag  % 详解: 条件判断：if (gradflag)
      calltype = 'fungrad';  % 详解: 赋值：计算表达式并保存到 calltype
   end  % 详解: 执行语句
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   
elseif isa(funstr, 'cell') & length(funstr)==2  % 详解: 条件判断：elseif (isa(funstr, 'cell') & length(funstr)==2)
   
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   [gradfcn, msg] = fcnchk(funstr{2},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   calltype = 'fun_then_grad';  % 详解: 赋值：计算表达式并保存到 calltype
   if ~gradflag  % 详解: 条件判断：if (~gradflag)
      warnstr = ...  % 详解: 赋值：计算表达式并保存到 warnstr
         sprintf('%s\n%s\n%s\n','Jacobian function provided but OPTIONS.Jacobian=''off'';', ...  % 详解: 执行语句
         '  ignoring Jacobian function and using finite-differencing.', ...  % 详解: 执行语句
         '  Rerun with OPTIONS.Jacobian=''on'' to use Jacobian function.');  % 详解: 执行语句
      warning(warnstr);  % 详解: 调用函数：warning(warnstr)
      calltype = 'fun';  % 详解: 赋值：计算表达式并保存到 calltype
   end  % 详解: 执行语句
   
elseif ~isa(funstr, 'cell')  % 详解: 条件判断：elseif (~isa(funstr, 'cell'))
   [funfcn, msg] = fcnchk(funstr,lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   if gradflag  % 详解: 条件判断：if (gradflag)
      gradfcn = funfcn;  % 详解: 赋值：计算表达式并保存到 gradfcn
   end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
   errmsg = sprintf('%s\n%s', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
      'FUN must be a function name, valid string expression, or inline object;', ...  % 详解: 执行语句
      ' or, FUN may be a cell array that contains these type of objects.');  % 详解: 执行语句
   error(errmsg)  % 详解: 调用函数：error(errmsg)
end  % 详解: 执行语句

allfcns{1} = calltype;  % 详解: 执行语句
allfcns{2} = caller;  % 详解: 执行语句
allfcns{3} = funfcn;  % 详解: 执行语句
allfcns{4} = gradfcn;  % 详解: 执行语句
allfcns{5}=[];  % 详解: 执行语句



