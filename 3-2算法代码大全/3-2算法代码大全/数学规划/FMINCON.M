% 文件: FMINCON.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(FUN,X,A,B,Aeq,Beq,LB,UB,NONLCON,options,varargin)  % 详解: 函数定义：fmincon(FUN,X,A,B,Aeq,Beq,LB,UB,NONLCON,options,varargin), 返回：X,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN


defaultopt = optimset('display','final','LargeScale','on', ...  % 详解: 赋值：将 optimset(...) 的结果保存到 defaultopt
   'TolX',1e-6,'TolFun',1e-6,'TolCon',1e-6,'DerivativeCheck','off',...  % 详解: 执行语句
   'Diagnostics','off',...  % 详解: 执行语句
   'GradObj','off','GradConstr','off','Hessian','off','MaxFunEvals','100*numberOfVariables',...  % 详解: 执行语句
   'DiffMaxChange',1e-1,'DiffMinChange',1e-8,...  % 详解: 执行语句
   'PrecondBandWidth',0,'TypicalX','ones(numberOfVariables,1)','MaxPCGIter','max(1,floor(numberOfVariables/2))', ...  % 详解: 创建全 1 矩阵/数组
   'TolPCG',0.1,'MaxIter',400,'HessPattern',[]);  % 详解: 执行语句
if nargin==1 & nargout <= 1 & isequal(FUN,'defaults')  % 详解: 条件判断：if (nargin==1 & nargout <= 1 & isequal(FUN,'defaults'))
   X = defaultopt;  % 详解: 赋值：计算表达式并保存到 X
   return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

large = 'large-scale';  % 详解: 赋值：计算表达式并保存到 large
medium = 'medium-scale';  % 详解: 赋值：计算表达式并保存到 medium

if nargin < 4, error('FMINCON requires at least four input arguments'); end  % 详解: 条件判断：if (nargin < 4, error('FMINCON requires at least four input arguments'); end)
if nargin < 10, options=[];  % 详解: 条件判断：if (nargin < 10, options=[];)
   if nargin < 9, NONLCON=[];  % 详解: 条件判断：if (nargin < 9, NONLCON=[];)
      if nargin < 8, UB = [];  % 详解: 条件判断：if (nargin < 8, UB = [];)
         if nargin < 7, LB = [];  % 详解: 条件判断：if (nargin < 7, LB = [];)
            if nargin < 6, Beq=[];  % 详解: 条件判断：if (nargin < 6, Beq=[];)
               if nargin < 5, Aeq =[];  % 详解: 条件判断：if (nargin < 5, Aeq =[];)
               end, end, end, end, end, end  % 详解: 执行语句
if isempty(NONLCON) & isempty(A) & isempty(Aeq) & isempty(UB) & isempty(LB)  % 详解: 条件判断：if (isempty(NONLCON) & isempty(A) & isempty(Aeq) & isempty(UB) & isempty(LB))
   error('FMINCON is for constrained problems. Use FMINUNC for unconstrained problems.')  % 详解: 调用函数：error('FMINCON is for constrained problems. Use FMINUNC for unconstrained problems.')
end  % 详解: 执行语句

if nargout > 4  % 详解: 条件判断：if (nargout > 4)
   computeLambda = 1;  % 详解: 赋值：计算表达式并保存到 computeLambda
else  % 详解: 条件判断：else 分支
   computeLambda = 0;  % 详解: 赋值：计算表达式并保存到 computeLambda
end  % 详解: 执行语句

caller='constr';  % 详解: 赋值：计算表达式并保存到 caller
lenVarIn = length(varargin);  % 详解: 赋值：将 length(...) 的结果保存到 lenVarIn
XOUT=X(:);  % 详解: 赋值：将 X(...) 的结果保存到 XOUT
numberOfVariables=length(XOUT);  % 详解: 赋值：将 length(...) 的结果保存到 numberOfVariables

options = optimset(defaultopt,options);  % 详解: 赋值：将 optimset(...) 的结果保存到 options
switch optimget(options,'display')  % 详解: 多分支选择：switch (optimget(options,'display'))
case {'off','none'}  % 详解: 分支：case {'off','none'}
   verbosity = 0;  % 详解: 赋值：计算表达式并保存到 verbosity
case 'iter'  % 详解: 分支：case 'iter'
   verbosity = 2;  % 详解: 赋值：计算表达式并保存到 verbosity
case 'final'  % 详解: 分支：case 'final'
   verbosity = 1;  % 详解: 赋值：计算表达式并保存到 verbosity
otherwise  % 详解: 默认分支：otherwise
   verbosity = 1;  % 详解: 赋值：计算表达式并保存到 verbosity
end  % 详解: 执行语句

B = B(:);  % 详解: 赋值：将 B(...) 的结果保存到 B
Beq = Beq(:);  % 详解: 赋值：将 Beq(...) 的结果保存到 Beq

[XOUT,l,u,msg] = checkbounds(XOUT,LB,UB,numberOfVariables);  % 详解: 执行语句
if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
   EXITFLAG = -1;  % 详解: 赋值：计算表达式并保存到 EXITFLAG
   [FVAL,OUTPUT,LAMBDA,GRAD,HESSIAN] = deal([]);  % 详解: 执行语句
   X(:)=XOUT;  % 详解: 执行语句
   if verbosity > 0  % 详解: 条件判断：if (verbosity > 0)
      disp(msg)  % 详解: 调用函数：disp(msg)
   end  % 详解: 执行语句
   return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句
lFinite = l(~isinf(l));  % 详解: 赋值：将 l(...) 的结果保存到 lFinite
uFinite = u(~isinf(u));  % 详解: 赋值：将 u(...) 的结果保存到 uFinite


meritFunctionType = 0;  % 详解: 赋值：计算表达式并保存到 meritFunctionType

diagnostics = isequal(optimget(options,'diagnostics','off'),'on');  % 详解: 赋值：将 isequal(...) 的结果保存到 diagnostics
gradflag =  strcmp(optimget(options,'GradObj'),'on');  % 详解: 赋值：将 strcmp(...) 的结果保存到 gradflag
hessflag = strcmp(optimget(options,'Hessian'),'on');  % 详解: 赋值：将 strcmp(...) 的结果保存到 hessflag
if isempty(NONLCON)  % 详解: 条件判断：if (isempty(NONLCON))
   constflag = 0;  % 详解: 赋值：计算表达式并保存到 constflag
else  % 详解: 条件判断：else 分支
   constflag = 1;  % 详解: 赋值：计算表达式并保存到 constflag
end  % 详解: 执行语句
gradconstflag =  strcmp(optimget(options,'GradConstr'),'on');  % 详解: 赋值：将 strcmp(...) 的结果保存到 gradconstflag
line_search = strcmp(optimget(options,'largescale','off'),'off');  % 详解: 赋值：将 strcmp(...) 的结果保存到 line_search

if ~isempty(FUN)  % 详解: 条件判断：if (~isempty(FUN))
   [funfcn, msg] = fprefcnchk(FUN,'fmincon',length(varargin),gradflag,hessflag);  % 详解: 获取向量/矩阵尺寸
else  % 详解: 条件判断：else 分支
   errmsg = sprintf('%s\n%s', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
      'FUN must be a function name, valid string expression, or inline object;', ...  % 详解: 执行语句
      ' or, FUN may be a cell array that contains these type of objects.');  % 详解: 执行语句
   error(errmsg)  % 详解: 调用函数：error(errmsg)
end  % 详解: 执行语句

if constflag  % 详解: 条件判断：if (constflag)
   [confcn, msg] = fprefcnchk(NONLCON,'fmincon',length(varargin),gradconstflag,[],1);  % 详解: 获取向量/矩阵尺寸
else  % 详解: 条件判断：else 分支
   confcn{1} = '';  % 详解: 执行语句
end  % 详解: 执行语句

[rowAeq,colAeq]=size(Aeq);  % 详解: 获取向量/矩阵尺寸
if ~line_search & isempty(NONLCON) & isempty(A) & isempty(Aeq) & gradflag  % 详解: 条件判断：if (~line_search & isempty(NONLCON) & isempty(A) & isempty(Aeq) & gradflag)
   OUTPUT.algorithm = large;  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
elseif ~line_search & isempty(NONLCON) & isempty(A) & isempty(lFinite) & isempty(uFinite) & gradflag ...  % 详解: 条件判断：elseif (~line_search & isempty(NONLCON) & isempty(A) & isempty(lFinite) & isempty(uFinite) & gradflag ...)
      & colAeq >= rowAeq  % 详解: 执行语句
   OUTPUT.algorithm = large;  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
elseif ~line_search  % 详解: 条件判断：elseif (~line_search)
   warning(['Trust region method does not currently solve this type of problem,',...  % 详解: 执行语句
         sprintf('\n'), 'switching to line search.'])  % 详解: 调用函数：sprintf('\n'), 'switching to line search.'])
   if isequal(funfcn{1},'fungradhess')  % 详解: 条件判断：if (isequal(funfcn{1},'fungradhess'))
      funfcn{1}='fungrad';  % 详解: 执行语句
      warning('Hessian provided by user will be ignored in line search algorithm')  % 详解: 调用函数：warning('Hessian provided by user will be ignored in line search algorithm')
      
   elseif  isequal(funfcn{1},'fun_then_grad_then_hess')  % 详解: 条件判断：elseif (isequal(funfcn{1},'fun_then_grad_then_hess'))
      funfcn{1}='fun_then_grad';  % 详解: 执行语句
      warning('Hessian provided by user will be ignored in line search algorithm')  % 详解: 调用函数：warning('Hessian provided by user will be ignored in line search algorithm')
   end  % 详解: 执行语句
   hessflag = 0;  % 详解: 赋值：计算表达式并保存到 hessflag
   OUTPUT.algorithm = medium;  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
elseif line_search  % 详解: 条件判断：elseif (line_search)
   OUTPUT.algorithm = medium;  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
   if issparse(Aeq) | issparse(A)  % 详解: 条件判断：if (issparse(Aeq) | issparse(A))
      warning('can not do sparse with line_search, converting to full')  % 详解: 调用函数：warning('can not do sparse with line_search, converting to full')
   end  % 详解: 执行语句
   
else  % 详解: 条件判断：else 分支
   error('Unrecognized combination of OPTIONS flags and calling sequence.')  % 详解: 调用函数：error('Unrecognized combination of OPTIONS flags and calling sequence.')
end  % 详解: 执行语句


lenvlb=length(l);  % 详解: 赋值：将 length(...) 的结果保存到 lenvlb
lenvub=length(u);  % 详解: 赋值：将 length(...) 的结果保存到 lenvub

if isequal(OUTPUT.algorithm,medium)  % 详解: 条件判断：if (isequal(OUTPUT.algorithm,medium))
   CHG = 1e-7*abs(XOUT)+1e-7*ones(numberOfVariables,1);  % 详解: 赋值：计算表达式并保存到 CHG
   i=1:lenvlb;  % 详解: 赋值：计算表达式并保存到 i
   lindex = XOUT(i)<l(i);  % 详解: 赋值：将 XOUT(...) 的结果保存到 lindex
   if any(lindex),  % 详解: 条件判断：if (any(lindex),)
      XOUT(lindex)=l(lindex)+1e-4;  % 详解: 执行语句
   end  % 详解: 执行语句
   i=1:lenvub;  % 详解: 赋值：计算表达式并保存到 i
   uindex = XOUT(i)>u(i);  % 详解: 赋值：将 XOUT(...) 的结果保存到 uindex
   if any(uindex)  % 详解: 条件判断：if (any(uindex))
      XOUT(uindex)=u(uindex);  % 详解: 调用函数：XOUT(uindex)=u(uindex)
      CHG(uindex)=-CHG(uindex);  % 详解: 调用函数：CHG(uindex)=-CHG(uindex)
   end  % 详解: 执行语句
   X(:) = XOUT;  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
   arg = (u >= 1e10); arg2 = (l <= -1e10);  % 详解: 赋值：计算表达式并保存到 arg
   u(arg) = inf*ones(length(arg(arg>0)),1);  % 详解: 调用函数：u(arg) = inf*ones(length(arg(arg>0)),1)
   l(arg2) = -inf*ones(length(arg2(arg2>0)),1);  % 详解: 调用函数：l(arg2) = -inf*ones(length(arg2(arg2>0)),1)
   if min(min(u-XOUT),min(XOUT-l)) < 0,  % 详解: 条件判断：if (min(min(u-XOUT),min(XOUT-l)) < 0,)
      XOUT = startx(u,l);  % 详解: 赋值：将 startx(...) 的结果保存到 XOUT
      X(:) = XOUT;  % 详解: 执行语句
   end  % 详解: 执行语句
end  % 详解: 执行语句

GRAD=zeros(numberOfVariables,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 GRAD
HESS = [];  % 详解: 赋值：计算表达式并保存到 HESS

switch funfcn{1}  % 详解: 多分支选择：switch (funfcn{1})
case 'fun'  % 详解: 分支：case 'fun'
   try  % 详解: 异常处理：try 块开始
      f = feval(funfcn{3},X,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied objective function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
case 'fungrad'  % 详解: 分支：case 'fungrad'
   try  % 详解: 异常处理：try 块开始
      [f,GRAD(:)] = feval(funfcn{3},X,varargin{:});  % 详解: 执行语句
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied objective function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
case 'fungradhess'  % 详解: 分支：case 'fungradhess'
   try  % 详解: 异常处理：try 块开始
      [f,GRAD(:),HESS] = feval(funfcn{3},X,varargin{:});  % 详解: 执行语句
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied objective function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
case 'fun_then_grad'  % 详解: 分支：case 'fun_then_grad'
   try  % 详解: 异常处理：try 块开始
      f = feval(funfcn{3},X,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied objective function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
   try  % 详解: 异常处理：try 块开始
      GRAD(:) = feval(funfcn{4},X,varargin{:});  % 详解: 调用函数：GRAD(:) = feval(funfcn{4},X,varargin{:})
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied objective gradient function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
case 'fun_then_grad_then_hess'  % 详解: 分支：case 'fun_then_grad_then_hess'
   try  % 详解: 异常处理：try 块开始
      f = feval(funfcn{3},X,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied objective function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
   try  % 详解: 异常处理：try 块开始
      GRAD(:) = feval(funfcn{4},X,varargin{:});  % 详解: 调用函数：GRAD(:) = feval(funfcn{4},X,varargin{:})
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied objective gradient function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
   try  % 详解: 异常处理：try 块开始
      HESS = feval(funfcn{5},X,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 HESS
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied objective Hessian function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
otherwise  % 详解: 默认分支：otherwise
   error('Undefined calltype in FMINCON');  % 详解: 调用函数：error('Undefined calltype in FMINCON')
end  % 详解: 执行语句

switch confcn{1}  % 详解: 多分支选择：switch (confcn{1})
case 'fun'  % 详解: 分支：case 'fun'
   try  % 详解: 异常处理：try 块开始
      [ctmp,ceqtmp] = feval(confcn{3},X,varargin{:});  % 详解: 执行语句
      c = ctmp(:); ceq = ceqtmp(:);  % 详解: 赋值：将 ctmp(...) 的结果保存到 c
      cGRAD = zeros(numberOfVariables,length(c));  % 详解: 赋值：将 zeros(...) 的结果保存到 cGRAD
      ceqGRAD = zeros(numberOfVariables,length(ceq));  % 详解: 赋值：将 zeros(...) 的结果保存到 ceqGRAD
   catch  % 详解: 异常处理：catch
      if findstr(xlate('Too many output arguments'),lasterr)  % 详解: 条件判断：if (findstr(xlate('Too many output arguments'),lasterr))
         if isa(confcn{3},'inline')  % 详解: 条件判断：if (isa(confcn{3},'inline'))
            errmsg = sprintf('%s%s%s\n%s\n%s\n%s', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
               'The inline function ',formula(confcn{3}),' representing the constraints',...  % 详解: 执行语句
               ' must return two outputs: the nonlinear inequality constraints and', ...  % 详解: 执行语句
               ' the nonlinear equality constraints.  At this time, inline objects may',...  % 详解: 执行语句
               ' only return one output argument: use an M-file function instead.');  % 详解: 执行语句
         else  % 详解: 条件判断：else 分支
            errmsg = sprintf('%s%s%s\n%s%s', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
               'The constraint function ',confcn{3},' must return two outputs:',...  % 详解: 执行语句
               ' the nonlinear inequality constraints and', ...  % 详解: 执行语句
               ' the nonlinear equality constraints.');  % 详解: 执行语句
         end  % 详解: 执行语句
         error(errmsg)  % 详解: 调用函数：error(errmsg)
      else  % 详解: 条件判断：else 分支
         errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
            'FMINCON cannot continue because user supplied nonlinear constraint function', ...  % 详解: 执行语句
            ' failed with the following error:', lasterr);  % 详解: 执行语句
         error(errmsg);  % 详解: 调用函数：error(errmsg)
      end  % 详解: 执行语句
   end  % 详解: 执行语句
   
case 'fungrad'  % 详解: 分支：case 'fungrad'
   try  % 详解: 异常处理：try 块开始
      [ctmp,ceqtmp,cGRAD,ceqGRAD] = feval(confcn{3},X,varargin{:});  % 详解: 执行语句
      c = ctmp(:); ceq = ceqtmp(:);  % 详解: 赋值：将 ctmp(...) 的结果保存到 c
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied nonlinear constraint function', ...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
case 'fun_then_grad'  % 详解: 分支：case 'fun_then_grad'
   try  % 详解: 异常处理：try 块开始
      [ctmp,ceqtmp] = feval(confcn{3},X,varargin{:});  % 详解: 执行语句
      c = ctmp(:); ceq = ceqtmp(:);  % 详解: 赋值：将 ctmp(...) 的结果保存到 c
      [cGRAD,ceqGRAD] = feval(confcn{4},X,varargin{:});  % 详解: 执行语句
   catch  % 详解: 异常处理：catch
      errmsg = sprintf('%s\n%s%s\n\n%s',...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
         'FMINCON cannot continue because user supplied nonlinear constraint function', ...  % 详解: 执行语句
         'or nonlinear constraint gradient function',...  % 详解: 执行语句
         ' failed with the following error:', lasterr);  % 详解: 执行语句
      error(errmsg);  % 详解: 调用函数：error(errmsg)
   end  % 详解: 执行语句
case ''  % 详解: 分支：case ''
   c=[]; ceq =[];  % 详解: 赋值：计算表达式并保存到 c
   cGRAD = zeros(numberOfVariables,length(c));  % 详解: 赋值：将 zeros(...) 的结果保存到 cGRAD
   ceqGRAD = zeros(numberOfVariables,length(ceq));  % 详解: 赋值：将 zeros(...) 的结果保存到 ceqGRAD
otherwise  % 详解: 默认分支：otherwise
   error('Undefined calltype in FMINCON');  % 详解: 调用函数：error('Undefined calltype in FMINCON')
end  % 详解: 执行语句

non_eq = length(ceq);  % 详解: 赋值：将 length(...) 的结果保存到 non_eq
non_ineq = length(c);  % 详解: 赋值：将 length(...) 的结果保存到 non_ineq
[lin_eq,Aeqcol] = size(Aeq);  % 详解: 获取向量/矩阵尺寸
[lin_ineq,Acol] = size(A);  % 详解: 获取向量/矩阵尺寸
[cgrow, cgcol]= size(cGRAD);  % 详解: 获取向量/矩阵尺寸
[ceqgrow, ceqgcol]= size(ceqGRAD);  % 详解: 获取向量/矩阵尺寸

eq = non_eq + lin_eq;  % 详解: 赋值：计算表达式并保存到 eq
ineq = non_ineq + lin_ineq;  % 详解: 赋值：计算表达式并保存到 ineq

if ~isempty(Aeq) & Aeqcol ~= numberOfVariables  % 详解: 条件判断：if (~isempty(Aeq) & Aeqcol ~= numberOfVariables)
   error('Aeq has the wrong number of columns.')  % 详解: 调用函数：error('Aeq has the wrong number of columns.')
end  % 详解: 执行语句
if ~isempty(A) & Acol ~= numberOfVariables  % 详解: 条件判断：if (~isempty(A) & Acol ~= numberOfVariables)
   error('A has the wrong number of columns.')  % 详解: 调用函数：error('A has the wrong number of columns.')
end  % 详解: 执行语句
if  cgrow~=numberOfVariables & cgcol~=non_ineq  % 详解: 条件判断：if (cgrow~=numberOfVariables & cgcol~=non_ineq)
   error('Gradient of the nonlinear inequality constraints is the wrong size.')  % 详解: 调用函数：error('Gradient of the nonlinear inequality constraints is the wrong size.')
end  % 详解: 执行语句
if ceqgrow~=numberOfVariables & ceqgcol~=non_eq  % 详解: 条件判断：if (ceqgrow~=numberOfVariables & ceqgcol~=non_eq)
   error('Gradient of the nonlinear equality constraints is the wrong size.')  % 详解: 调用函数：error('Gradient of the nonlinear equality constraints is the wrong size.')
end  % 详解: 执行语句

if diagnostics > 0  % 详解: 条件判断：if (diagnostics > 0)
   msg = diagnose('fmincon',OUTPUT,gradflag,hessflag,constflag,gradconstflag,...  % 详解: 赋值：将 diagnose(...) 的结果保存到 msg
      line_search,options,XOUT,non_eq,...  % 详解: 执行语句
      non_ineq,lin_eq,lin_ineq,l,u,funfcn,confcn,f,GRAD,HESS,c,ceq,cGRAD,ceqGRAD);  % 详解: 执行语句
end  % 详解: 执行语句


if isequal(OUTPUT.algorithm,medium)  % 详解: 条件判断：if (isequal(OUTPUT.algorithm,medium))
   [X,FVAL,lambda,EXITFLAG,OUTPUT,GRAD,HESSIAN]=...  % 详解: 执行语句
      nlconst(funfcn,X,l,u,full(A),B,full(Aeq),Beq,confcn,options, ...  % 详解: 执行语句
      verbosity,gradflag,gradconstflag,hessflag,meritFunctionType,...  % 详解: 执行语句
      CHG,f,GRAD,HESS,c,ceq,cGRAD,ceqGRAD,varargin{:});  % 详解: 执行语句
   LAMBDA=lambda;  % 详解: 赋值：计算表达式并保存到 LAMBDA
   
   
else  % 详解: 条件判断：else 分支
   if (isequal(funfcn{1}, 'fun_then_grad_then_hess') | isequal(funfcn{1}, 'fungradhess'))  % 详解: 条件判断：if ((isequal(funfcn{1}, 'fun_then_grad_then_hess') | isequal(funfcn{1}, 'fungradhess')))
      Hstr=[];  % 详解: 赋值：计算表达式并保存到 Hstr
   elseif (isequal(funfcn{1}, 'fun_then_grad') | isequal(funfcn{1}, 'fungrad'))  % 详解: 条件判断：elseif ((isequal(funfcn{1}, 'fun_then_grad') | isequal(funfcn{1}, 'fungrad')))
      n = length(XOUT);  % 详解: 赋值：将 length(...) 的结果保存到 n
      Hstr = optimget(options,'HessPattern',[]);  % 详解: 赋值：将 optimget(...) 的结果保存到 Hstr
      if isempty(Hstr)  % 详解: 条件判断：if (isempty(Hstr))
         Hstr = sparse(ones(n));  % 详解: 赋值：将 sparse(...) 的结果保存到 Hstr
      end  % 详解: 执行语句
      if ischar(Hstr)  % 详解: 条件判断：if (ischar(Hstr))
         Hstr = eval(Hstr);  % 详解: 赋值：将 eval(...) 的结果保存到 Hstr
      end  % 详解: 执行语句
   end  % 详解: 执行语句
   
   if isempty(Aeq)  % 详解: 条件判断：if (isempty(Aeq))
      [X,FVAL,LAMBDA,EXITFLAG,OUTPUT,GRAD,HESSIAN] = ...  % 详解: 执行语句
         sfminbx(funfcn,X,l,u,verbosity,options,computeLambda,f,GRAD,HESS,Hstr,varargin{:});  % 详解: 调用函数：sfminbx(funfcn,X,l,u,verbosity,options,computeLambda,f,GRAD,HESS,Hstr,varargin{:})
   else  % 详解: 条件判断：else 分支
      [X,FVAL,LAMBDA,EXITFLAG,OUTPUT,GRAD,HESSIAN] = ...  % 详解: 执行语句
         sfminle(funfcn,X,sparse(Aeq),Beq,verbosity,options,computeLambda,f,GRAD,HESS,Hstr,varargin{:});  % 详解: 调用函数：sfminle(funfcn,X,sparse(Aeq),Beq,verbosity,options,computeLambda,f,GRAD,HESS,Hstr,varargin{:})
   end  % 详解: 执行语句
end  % 详解: 执行语句


function [allfcns,msg] = fprefcnchk(funstr,caller,lenVarIn,gradflag,hessflag,constrflag)  % 详解: 函数定义：fprefcnchk(funstr,caller,lenVarIn,gradflag,hessflag,constrflag), 返回：allfcns,msg
if nargin < 6  % 详解: 条件判断：if (nargin < 6)
   constrflag = 0;  % 详解: 赋值：计算表达式并保存到 constrflag
end  % 详解: 执行语句
if constrflag  % 详解: 条件判断：if (constrflag)
   graderrmsg = 'Constraint gradient function expected (OPTIONS.GradConstr==''on'') but not found.';  % 详解: 赋值：计算表达式并保存到 graderrmsg
   warnstr = ...  % 详解: 赋值：计算表达式并保存到 warnstr
      sprintf('%s\n%s\n%s\n','Constraint gradient function provided but OPTIONS.GradConstr==''off'';', ...  % 详解: 执行语句
      '  ignoring constraint gradient function and using finite-differencing.', ...  % 详解: 执行语句
      '  Rerun with OPTIONS.GradConstr==''on'' to use constraint gradient function.');  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
   graderrmsg = 'Gradient function expected OPTIONS.GradObj==''on'' but not found.';  % 详解: 赋值：计算表达式并保存到 graderrmsg
   warnstr = ...  % 详解: 赋值：计算表达式并保存到 warnstr
      sprintf('%s\n%s\n%s\n','Gradient function provided but OPTIONS.GradObj==''off'';', ...  % 详解: 执行语句
      '  ignoring gradient function and using finite-differencing.', ...  % 详解: 执行语句
      '  Rerun with OPTIONS.GradObj==''on'' to use gradient function.');  % 详解: 执行语句
   
end  % 详解: 执行语句
msg='';  % 详解: 赋值：计算表达式并保存到 msg
allfcns = {};  % 详解: 赋值：计算表达式并保存到 allfcns
funfcn = [];  % 详解: 赋值：计算表达式并保存到 funfcn
gradfcn = [];  % 详解: 赋值：计算表达式并保存到 gradfcn
hessfcn = [];  % 详解: 赋值：计算表达式并保存到 hessfcn
if gradflag & hessflag  % 详解: 条件判断：if (gradflag & hessflag)
   calltype = 'fungradhess';  % 详解: 赋值：计算表达式并保存到 calltype
elseif gradflag  % 详解: 条件判断：elseif (gradflag)
   calltype = 'fungrad';  % 详解: 赋值：计算表达式并保存到 calltype
else  % 详解: 条件判断：else 分支
   calltype = 'fun';  % 详解: 赋值：计算表达式并保存到 calltype
end  % 详解: 执行语句

if isa(funstr, 'cell') & length(funstr)==1  % 详解: 条件判断：if (isa(funstr, 'cell') & length(funstr)==1)
   if gradflag  % 详解: 条件判断：if (gradflag)
      error(graderrmsg)  % 详解: 调用函数：error(graderrmsg)
   end  % 详解: 执行语句
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   
elseif isa(funstr, 'cell') & length(funstr)==2 & isempty(funstr{2})  % 详解: 条件判断：elseif (isa(funstr, 'cell') & length(funstr)==2 & isempty(funstr{2}))
   if gradflag  % 详解: 条件判断：if (gradflag)
      error(graderrmsg)  % 详解: 调用函数：error(graderrmsg)
   end  % 详解: 执行语句
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   
elseif isa(funstr, 'cell') & length(funstr)==2  % 详解: 条件判断：elseif (isa(funstr, 'cell') & length(funstr)==2)
   
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   [gradfcn, msg] = fcnchk(funstr{2},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   calltype = 'fun_then_grad';  % 详解: 赋值：计算表达式并保存到 calltype
   if ~gradflag  % 详解: 条件判断：if (~gradflag)
      warning(warnstr);  % 详解: 调用函数：warning(warnstr)
      calltype = 'fun';  % 详解: 赋值：计算表达式并保存到 calltype
   end  % 详解: 执行语句
   
   
elseif isa(funstr, 'cell') & length(funstr)==3 ...  % 详解: 条件判断：elseif (isa(funstr, 'cell') & length(funstr)==3 ...)
      & ~isempty(funstr{1}) & isempty(funstr{2}) & isempty(funstr{3})  % 详解: 执行语句
   if gradflag  % 详解: 条件判断：if (gradflag)
      error(graderrmsg)  % 详解: 调用函数：error(graderrmsg)
   end  % 详解: 执行语句
   if hessflag  % 详解: 条件判断：if (hessflag)
      error('Hessian function expected but not found.')  % 详解: 调用函数：error('Hessian function expected but not found.')
   end  % 详解: 执行语句
   
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   
elseif isa(funstr, 'cell') & length(funstr)==3 ...  % 详解: 条件判断：elseif (isa(funstr, 'cell') & length(funstr)==3 ...)
      & ~isempty(funstr{2}) & ~isempty(funstr{3})  % 详解: 执行语句
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   [gradfcn, msg] = fcnchk(funstr{2},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   [hessfcn, msg] = fcnchk(funstr{3},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   calltype = 'fun_then_grad_then_hess';  % 详解: 赋值：计算表达式并保存到 calltype
   if ~hessflag & ~gradflag  % 详解: 条件判断：if (~hessflag & ~gradflag)
      hwarnstr = sprintf('%s\n%s\n%s\n','Hessian and gradient functions provided ', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 hwarnstr
         '  but OPTIONS.Hessian==''off'' and OPTIONS.GradObj==''off''; ignoring Hessian and gradient functions.', ...  % 详解: 执行语句
         '  Rerun with OPTIONS.Hessian==''on'' and OPTIONS.GradObj==''on'' to use derivative functions.');  % 详解: 执行语句
      warning(hwarnstr)  % 详解: 调用函数：warning(hwarnstr)
      calltype = 'fun';  % 详解: 赋值：计算表达式并保存到 calltype
   elseif hessflag & ~gradflag  % 详解: 条件判断：elseif (hessflag & ~gradflag)
      warnstr = ...  % 详解: 赋值：计算表达式并保存到 warnstr
         sprintf('%s\n%s\n%s\n','Hessian and gradient functions provided ', ...  % 详解: 执行语句
         '  but OPTIONS.GradObj==''off''; ignoring Hessian and gradient functions.', ...  % 详解: 执行语句
         '  Rerun with OPTIONS.Hessian==''on'' and OPTIONS.GradObj==''on'' to use derivative functions.');  % 详解: 执行语句
      warning(warnstr)  % 详解: 调用函数：warning(warnstr)
      calltype = 'fun';  % 详解: 赋值：计算表达式并保存到 calltype
   elseif ~hessflag & gradflag  % 详解: 条件判断：elseif (~hessflag & gradflag)
      hwarnstr = ...  % 详解: 赋值：计算表达式并保存到 hwarnstr
         sprintf('%s\n%s\n%s\n','Hessian function provided but OPTIONS.Hessian==''off'';', ...  % 详解: 执行语句
         '  ignoring Hessian function,', ...  % 详解: 执行语句
         '  Rerun with OPTIONS.Hessian==''on'' to use Hessian function.');  % 详解: 执行语句
      warning(hwarnstr);  % 详解: 调用函数：warning(hwarnstr)
      calltype = 'fun_then_grad';  % 详解: 赋值：计算表达式并保存到 calltype
   end  % 详解: 执行语句
   
elseif isa(funstr, 'cell') & length(funstr)==3 ...  % 详解: 条件判断：elseif (isa(funstr, 'cell') & length(funstr)==3 ...)
      & ~isempty(funstr{2}) & isempty(funstr{3})  % 详解: 执行语句
   if hessflag  % 详解: 条件判断：if (hessflag)
      error('Hessian function expected but not found.')  % 详解: 调用函数：error('Hessian function expected but not found.')
   end  % 详解: 执行语句
   [funfcn, msg] = fcnchk(funstr{1},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   [gradfcn, msg] = fcnchk(funstr{2},lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   calltype = 'fun_then_grad';  % 详解: 赋值：计算表达式并保存到 calltype
   if ~gradflag  % 详解: 条件判断：if (~gradflag)
      warning(warnstr);  % 详解: 调用函数：warning(warnstr)
      calltype = 'fun';  % 详解: 赋值：计算表达式并保存到 calltype
   end  % 详解: 执行语句
   
elseif isa(funstr, 'cell') & length(funstr)==3 ...  % 详解: 条件判断：elseif (isa(funstr, 'cell') & length(funstr)==3 ...)
      & isempty(funstr{2}) & ~isempty(funstr{3})  % 详解: 执行语句
   error('Hessian function given without gradient function.')  % 详解: 调用函数：error('Hessian function given without gradient function.')
   
elseif ~isa(funstr, 'cell')  % 详解: 条件判断：elseif (~isa(funstr, 'cell'))
   [funfcn, msg] = fcnchk(funstr,lenVarIn);  % 详解: 执行语句
   if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
      if constrflag  % 详解: 条件判断：if (constrflag)
         msg = ['NONLCON must be a function name.'];  % 详解: 赋值：计算表达式并保存到 msg
      end  % 详解: 执行语句
      
      error(msg);  % 详解: 调用函数：error(msg)
   end  % 详解: 执行语句
   if gradflag  % 详解: 条件判断：if (gradflag)
      gradfcn = funfcn;  % 详解: 赋值：计算表达式并保存到 gradfcn
   end  % 详解: 执行语句
   if hessflag & ~gradflag  % 详解: 条件判断：if (hessflag & ~gradflag)
      hwarnstr = ...  % 详解: 赋值：计算表达式并保存到 hwarnstr
         sprintf('%s\n%s\n%s\n','OPTIONS.Hessian==''on'' ', ...  % 详解: 执行语句
         '  but OPTIONS.GradObj==''off''; ignoring Hessian and gradient functions.', ...  % 详解: 执行语句
         '  Rerun with OPTIONS.Hessian==''on'' and OPTIONS.GradObj==''on'' to use derivative functions.');  % 详解: 执行语句
      warning(hwarnstr)  % 详解: 调用函数：warning(hwarnstr)
   end  % 详解: 执行语句
   
else  % 详解: 条件判断：else 分支
   errmsg = sprintf('%s\n%s', ...  % 详解: 赋值：将 sprintf(...) 的结果保存到 errmsg
      'FUN must be a function name or inline object;', ...  % 详解: 执行语句
      ' or, FUN may be a cell array that contains these type of objects.');  % 详解: 执行语句
   error(errmsg)  % 详解: 调用函数：error(errmsg)
end  % 详解: 执行语句

allfcns{1} = calltype;  % 详解: 执行语句
allfcns{2} = caller;  % 详解: 执行语句
allfcns{3} = funfcn;  % 详解: 执行语句
allfcns{4} = gradfcn;  % 详解: 执行语句
allfcns{5} = hessfcn;  % 详解: 执行语句





