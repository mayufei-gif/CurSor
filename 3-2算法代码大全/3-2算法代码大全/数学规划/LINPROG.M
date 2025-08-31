% 文件: LINPROG.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x,fval,exitflag,output,lambda]=linprog(f,A,B,Aeq,Beq,lb,ub,x0,options)  % 详解: 函数定义：linprog(f,A,B,Aeq,Beq,lb,ub,x0,options), 返回：x,fval,exitflag,output,lambda


defaultopt = optimset('display','final',...  % 详解: 赋值：将 optimset(...) 的结果保存到 defaultopt
   'TolFun',1e-8,'Diagnostics','off',...  % 详解: 执行语句
   'LargeScale','on','maxiter',85);  % 详解: 执行语句

if nargin==1 & nargout <= 1 & isequal(f,'defaults')  % 详解: 条件判断：if (nargin==1 & nargout <= 1 & isequal(f,'defaults'))
   x = defaultopt;  % 详解: 赋值：计算表达式并保存到 x
   return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

if nargin < 9, options = [];  % 详解: 条件判断：if (nargin < 9, options = [];)
   if nargin < 8, x0 = [];  % 详解: 条件判断：if (nargin < 8, x0 = [];)
      if nargin < 7, ub = [];  % 详解: 条件判断：if (nargin < 7, ub = [];)
         if nargin < 6, lb = [];  % 详解: 条件判断：if (nargin < 6, lb = [];)
            if nargin < 5, Beq = [];  % 详解: 条件判断：if (nargin < 5, Beq = [];)
               if nargin < 4, Aeq = [];  % 详解: 条件判断：if (nargin < 4, Aeq = [];)
               end, end, end, end, end, end  % 详解: 执行语句
if nargout > 4  % 详解: 条件判断：if (nargout > 4)
   computeLambda = 1;  % 详解: 赋值：计算表达式并保存到 computeLambda
else  % 详解: 条件判断：else 分支
   computeLambda = 0;  % 详解: 赋值：计算表达式并保存到 computeLambda
end  % 详解: 执行语句

options = optimset(defaultopt,options);  % 详解: 赋值：将 optimset(...) 的结果保存到 options
largescale = isequal(optimget(options,'largescale'),'on');  % 详解: 赋值：将 isequal(...) 的结果保存到 largescale
diagnostics = isequal(optimget(options,'diagnostics','off'),'on');  % 详解: 赋值：将 isequal(...) 的结果保存到 diagnostics
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

[nineqcstr,nvars]=size(A);  % 详解: 获取向量/矩阵尺寸
[neqcstr, nvarseq]=size(Aeq);  % 详解: 获取向量/矩阵尺寸
nvars = max([length(f),nvars,nvarseq]);  % 详解: 赋值：将 max(...) 的结果保存到 nvars
ncstr = nineqcstr + neqcstr;  % 详解: 赋值：计算表达式并保存到 ncstr

if isempty(A), A=zeros(0,nvars); end  % 详解: 条件判断：if (isempty(A), A=zeros(0,nvars); end)
if isempty(B), B=zeros(0,1); end  % 详解: 条件判断：if (isempty(B), B=zeros(0,1); end)
if isempty(Aeq), Aeq=zeros(0,nvars); end  % 详解: 条件判断：if (isempty(Aeq), Aeq=zeros(0,nvars); end)
if isempty(Beq), Beq=zeros(0,1); end  % 详解: 条件判断：if (isempty(Beq), Beq=zeros(0,1); end)

f = f(:);  % 详解: 赋值：将 f(...) 的结果保存到 f
B = B(:);  % 详解: 赋值：将 B(...) 的结果保存到 B
Beq = Beq(:);  % 详解: 赋值：将 Beq(...) 的结果保存到 Beq

[x0,lb,ub,msg] = checkbounds(x0,lb,ub,nvars);  % 详解: 执行语句
if ~isempty(msg)  % 详解: 条件判断：if (~isempty(msg))
   exitflag = -1;  % 详解: 赋值：计算表达式并保存到 exitflag
   output = []; x=x0; fval = []; lambda = [];  % 详解: 赋值：计算表达式并保存到 output
   if verbosity > 0  % 详解: 条件判断：if (verbosity > 0)
      disp(msg)  % 详解: 调用函数：disp(msg)
   end  % 详解: 执行语句
   return  % 详解: 返回：从当前函数返回
end  % 详解: 执行语句

caller = 'linprog';  % 详解: 赋值：计算表达式并保存到 caller
ncstr = nineqcstr + neqcstr;  % 详解: 赋值：计算表达式并保存到 ncstr

if largescale  % 详解: 条件判断：if (largescale)
   OUTPUT.algorithm = 'large-scale: interior point';  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
else  % 详解: 条件判断：else 分支
   OUTPUT.algorithm  = 'medium-scale: active-set';  % 详解: 赋值：计算表达式并保存到 OUTPUT.algorithm
end  % 详解: 执行语句

if diagnostics  % 详解: 条件判断：if (diagnostics)
   gradflag = []; hessflag = []; line_search=[];  % 详解: 赋值：计算表达式并保存到 gradflag
   constflag = 0; gradconstflag = 0; non_eq=0;non_ineq=0;  % 详解: 赋值：计算表达式并保存到 constflag
   lin_eq=size(Aeq,1); lin_ineq=size(A,1); XOUT=ones(nvars,1);  % 详解: 赋值：将 size(...) 的结果保存到 lin_eq
   funfcn{1} = [];ff=[]; GRAD=[];HESS=[];  % 详解: 执行语句
   confcn{1}=[];c=[];ceq=[];cGRAD=[];ceqGRAD=[];  % 详解: 执行语句
   msg = diagnose('linprog',OUTPUT,gradflag,hessflag,constflag,gradconstflag,...  % 详解: 赋值：将 diagnose(...) 的结果保存到 msg
      line_search,options,XOUT,non_eq,...  % 详解: 执行语句
      non_ineq,lin_eq,lin_ineq,lb,ub,funfcn,confcn,ff,GRAD,HESS,c,ceq,cGRAD,ceqGRAD);  % 详解: 执行语句
end  % 详解: 执行语句

if (largescale)  % 详解: 条件判断：if ((largescale))
   if ~isempty(x0) & verbosity > 0  % 详解: 条件判断：if (~isempty(x0) & verbosity > 0)
      warning('Interior Point method is ignoring starting point')  % 详解: 调用函数：warning('Interior Point method is ignoring starting point')
   end  % 详解: 执行语句
   [x,fval,lambda,exitflag,output] = lipsol(f,A,B,Aeq,Beq,lb,ub,options,computeLambda);  % 详解: 执行语句
   output.algorithm = 'lipsol';  % 详解: 赋值：计算表达式并保存到 output.algorithm
else  % 详解: 条件判断：else 分支
   if ~largescale  & (issparse(A) | issparse(Aeq) )  % 详解: 条件判断：if (~largescale  & (issparse(A) | issparse(Aeq) ))
      if verbosity > 0  % 详解: 条件判断：if (verbosity > 0)
         disp('The medium-scale (active-set) algorithm does not currently handle sparse matrices.')  % 详解: 调用函数：disp('The medium-scale (active-set) algorithm does not currently handle sparse matrices.')
         disp('Converting to full matrices to solve.')  % 详解: 调用函数：disp('Converting to full matrices to solve.')
      end  % 详解: 执行语句
   end  % 详解: 执行语句
   if isempty(x0), x0=zeros(nvars,1); end  % 详解: 条件判断：if (isempty(x0), x0=zeros(nvars,1); end)
   [x,lambdaqp,exitflag,output]= ...  % 详解: 执行语句
      qpsub([],full(f),full([Aeq;A]),full([Beq;B]),lb,ub,x0,neqcstr,verbosity,caller,ncstr,nvars,options);  % 详解: 调用函数：qpsub([],full(f),full([Aeq;A]),full([Beq;B]),lb,ub,x0,neqcstr,verbosity,caller,ncstr,nvars,options)
   output.algorithm = 'medium-scale: activeset';  % 详解: 赋值：计算表达式并保存到 output.algorithm
end  % 详解: 执行语句

if isequal(output.algorithm , 'medium-scale: activeset')  % 详解: 条件判断：if (isequal(output.algorithm , 'medium-scale: activeset'))
   fval = f'*x;   % 赋值：设置变量 fval  % 详解: 赋值：计算表达式并保存到 fval  % 详解: 赋值：计算表达式并保存到 fval
   llb = length(lb);  % 详解: 赋值：将 length(...) 的结果保存到 llb
   lub = length(ub);  % 详解: 赋值：将 length(...) 的结果保存到 lub
   lambda.lower = zeros(llb,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 lambda.lower
   lambda.upper = zeros(lub,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 lambda.upper
   arglb = ~isinf(lb); lenarglb = nnz(arglb);  % 详解: 赋值：计算表达式并保存到 arglb
   argub = ~isinf(ub); lenargub = nnz(argub);  % 详解: 赋值：计算表达式并保存到 argub
   lambda.eqlin = lambdaqp(1:neqcstr,1);  % 详解: 赋值：将 lambdaqp(...) 的结果保存到 lambda.eqlin
   lambda.ineqlin = lambdaqp(neqcstr+1:neqcstr+nineqcstr,1);  % 详解: 赋值：将 lambdaqp(...) 的结果保存到 lambda.ineqlin
   lambda.lower(arglb) = lambdaqp(neqcstr+nineqcstr+1:neqcstr+nineqcstr+lenarglb);  % 详解: 执行语句
   lambda.upper(argub) = lambdaqp(neqcstr+nineqcstr+lenarglb+1:neqcstr+nineqcstr+lenarglb+lenargub);  % 详解: 执行语句
      
   output.firstorderopt=[];  % 详解: 赋值：计算表达式并保存到 output.firstorderopt
   output.cgiterations =[];  % 详解: 赋值：计算表达式并保存到 output.cgiterations
   
   if verbosity > 0  % 详解: 条件判断：if (verbosity > 0)
      if ( exitflag ==1 )  % 详解: 条件判断：if (( exitflag ==1 ))
         disp('Optimization terminated successfully.');  % 详解: 调用函数：disp('Optimization terminated successfully.')
      end  % 详解: 执行语句
      if ( exitflag == 2)  % 详解: 条件判断：if (( exitflag == 2))
         disp('Optimization completed.');  % 详解: 调用函数：disp('Optimization completed.')
      end  % 详解: 执行语句
      if (exitflag ==0)  % 详解: 条件判断：if ((exitflag ==0))
         disp('Maximum number of iterations exceeded;')  % 详解: 调用函数：disp('Maximum number of iterations exceeded;')
         disp('   increase options.MaxIter')  % 详解: 调用函数：disp(' increase options.MaxIter')
      end  % 详解: 执行语句

   end  % 详解: 执行语句
end  % 详解: 执行语句




