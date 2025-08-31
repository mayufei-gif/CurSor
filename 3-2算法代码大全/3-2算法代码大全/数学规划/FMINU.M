% 文件: FMINU.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x,OPTIONS] = fminu(FUN,x,OPTIONS,GRADFUN,varargin)  % 详解: 函数定义：fminu(FUN,x,OPTIONS,GRADFUN,varargin), 返回：x,OPTIONS



XOUT=x(:);  % 详解: 赋值：将 x(...) 的结果保存到 XOUT
nvars=length(XOUT);  % 详解: 赋值：将 length(...) 的结果保存到 nvars

if nargin < 2, error('fminu requires two input arguments');end  % 详解: 条件判断：if (nargin < 2, error('fminu requires two input arguments');end)
if nargin < 3, OPTIONS=[]; end  % 详解: 条件判断：if (nargin < 3, OPTIONS=[]; end)
if nargin < 4, GRADFUN=[]; end  % 详解: 条件判断：if (nargin < 4, GRADFUN=[]; end)

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

f = feval(funfcn,x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
n = length(XOUT);  % 详解: 赋值：将 length(...) 的结果保存到 n
GRAD=zeros(nvars,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 GRAD
OLDX=XOUT;  % 详解: 赋值：计算表达式并保存到 OLDX
MATX=zeros(3,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 MATX
MATL=[f;0;0];  % 详解: 赋值：计算表达式并保存到 MATL
OLDF=f;  % 详解: 赋值：计算表达式并保存到 OLDF
FIRSTF=f;  % 详解: 赋值：计算表达式并保存到 FIRSTF
[OLDX,OLDF,HESS,OPTIONS]=optint(XOUT,f,OPTIONS);  % 详解: 执行语句
CHG = 1e-7*abs(XOUT)+1e-7*ones(nvars,1);  % 详解: 赋值：计算表达式并保存到 CHG
SD = zeros(nvars,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 SD
diff = zeros(nvars,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 diff
PCNT = 0;  % 详解: 赋值：计算表达式并保存到 PCNT
how = '';  % 详解: 赋值：计算表达式并保存到 how


OPTIONS(10)=2;  % 详解: 执行语句
status =-1;  % 详解: 赋值：计算表达式并保存到 status

while status ~= 1  % 详解: while 循环：当 (status ~= 1) 为真时迭代
    if isempty(gradfcn) | OPTIONS(9)  % 详解: 条件判断：if (isempty(gradfcn) | OPTIONS(9))
        OLDF=f;  % 详解: 赋值：计算表达式并保存到 OLDF
        f = find(10*abs(CHG)>abs(SD));  % 详解: 赋值：将 find(...) 的结果保存到 f
        CHG(f) = -0.1*SD(f);  % 详解: 调用函数：CHG(f) = -0.1*SD(f)
        CHG = sign(CHG+eps).*min(max(abs(CHG),OPTIONS(16)),OPTIONS(17));  % 详解: 赋值：将 sign(...) 的结果保存到 CHG
        for gcnt=1:nvars  % 详解: for 循环：迭代变量 gcnt 遍历 1:nvars
            XOUT(gcnt,1)=XOUT(gcnt)+CHG(gcnt);  % 详解: 调用函数：XOUT(gcnt,1)=XOUT(gcnt)+CHG(gcnt)
            x(:) = XOUT;  % 详解: 执行语句
            f = feval(funfcn,x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
            GRAD(gcnt)=(f-OLDF)/(CHG(gcnt));  % 详解: 调用函数：GRAD(gcnt)=(f-OLDF)/(CHG(gcnt))
            if f < OLDF  % 详解: 条件判断：if (f < OLDF)
                OLDF=f;  % 详解: 赋值：计算表达式并保存到 OLDF
            else  % 详解: 条件判断：else 分支
                XOUT(gcnt)=XOUT(gcnt)-CHG(gcnt);  % 详解: 调用函数：XOUT(gcnt)=XOUT(gcnt)-CHG(gcnt)
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        CHG = 1e-8./(GRAD + eps);  % 详解: 赋值：计算表达式并保存到 CHG
        f = OLDF;  % 详解: 赋值：计算表达式并保存到 f
        OPTIONS(10)=OPTIONS(10)+nvars;  % 详解: 执行语句
        if OPTIONS(9) == 1  % 详解: 条件判断：if (OPTIONS(9) == 1)
            GRADFD = GRAD;  % 详解: 赋值：计算表达式并保存到 GRADFD
            x(:)=XOUT;  % 详解: 执行语句
            GRAD(:) = feval(gradfcn,x,varargin{:});  % 详解: 调用函数：GRAD(:) = feval(gradfcn,x,varargin{:})
            if isa(gradfcn,'inline')  % 详解: 条件判断：if (isa(gradfcn,'inline'))
              graderr(GRADFD, GRAD, formula(gradfcn));  % 详解: 调用函数：graderr(GRADFD, GRAD, formula(gradfcn))
            else  % 详解: 条件判断：else 分支
              graderr(GRADFD, GRAD,  gradfcn);  % 详解: 调用函数：graderr(GRADFD, GRAD, gradfcn)
            end  % 详解: 执行语句
             OPTIONS(9) = 0;  % 详解: 执行语句
        end  % 详解: 执行语句
                
    else  % 详解: 条件判断：else 分支
        OPTIONS(11)=OPTIONS(11)+1;  % 详解: 执行语句
        x(:)=XOUT;  % 详解: 执行语句
        GRAD(:) = feval(gradfcn,x,varargin{:});  % 详解: 调用函数：GRAD(:) = feval(gradfcn,x,varargin{:})
    end  % 详解: 执行语句
if status == -1  % 详解: 条件判断：if (status == -1)
    SD=-GRAD;  % 详解: 赋值：计算表达式并保存到 SD
    FIRSTF=f;  % 详解: 赋值：计算表达式并保存到 FIRSTF
    OLDG=GRAD;  % 详解: 赋值：计算表达式并保存到 OLDG
    GDOLD=GRAD'*SD;  % 赋值：设置变量 GDOLD  % 详解: 赋值：计算表达式并保存到 GDOLD  % 详解: 赋值：计算表达式并保存到 GDOLD
    OPTIONS(18) = max(0.001, min([1,2*abs(f/GDOLD)]));  % 详解: 调用函数：OPTIONS(18) = max(0.001, min([1,2*abs(f/GDOLD)]))
    if OPTIONS(1)>0  % 详解: 条件判断：if (OPTIONS(1)>0)
        disp([sprintf('%5.0f %12.6g %12.6g ',OPTIONS(10),f,OPTIONS(18)),sprintf('%12.3g  ',GDOLD)]);  % 详解: 调用函数：disp([sprintf('%5.0f %12.6g %12.6g ',OPTIONS(10),f,OPTIONS(18)),sprintf('%12.3g ',GDOLD)])
    end  % 详解: 执行语句
    XOUT=XOUT+OPTIONS(18)*SD;  % 详解: 赋值：计算表达式并保存到 XOUT
    status=4;  % 详解: 赋值：计算表达式并保存到 status
    if OPTIONS(7)==0; PCNT=1; end  % 详解: 条件判断：if (OPTIONS(7)==0; PCNT=1; end)
         
else  % 详解: 条件判断：else 分支
    gdnew=GRAD'*SD;  % 赋值：设置变量 gdnew  % 详解: 赋值：计算表达式并保存到 gdnew  % 详解: 赋值：计算表达式并保存到 gdnew
    if OPTIONS(1)>0,  % 详解: 条件判断：if (OPTIONS(1)>0,)
        num=[sprintf('%5.0f %12.6g %12.6g ',OPTIONS(10),f,OPTIONS(18)),sprintf('%12.3g  ',gdnew)];  % 详解: 赋值：计算表达式并保存到 num
    end  % 详解: 执行语句
    if (gdnew>0 & f>FIRSTF)|~finite(f)  % 详解: 条件判断：if ((gdnew>0 & f>FIRSTF)|~finite(f))
        how='inter';  % 详解: 赋值：计算表达式并保存到 how
        [stepsize]=cubici1(f,FIRSTF,gdnew,GDOLD,OPTIONS(18));  % 详解: 执行语句
        if stepsize<0|isnan(stepsize), stepsize=OPTIONS(18)/2; how='C1f '; end  % 详解: 条件判断：if (stepsize<0|isnan(stepsize), stepsize=OPTIONS(18)/2; how='C1f '; end)
        if OPTIONS(18)<0.1&OPTIONS(6)==0  % 详解: 条件判断：if (OPTIONS(18)<0.1&OPTIONS(6)==0)
            if stepsize*norm(SD)<eps  % 详解: 条件判断：if (stepsize*norm(SD)<eps)
                stepsize=exp(rand(1,1)-1)-0.1;  % 详解: 赋值：将 exp(...) 的结果保存到 stepsize
                how='RANDOM STEPLENGTH';  % 详解: 赋值：计算表达式并保存到 how
                status=0;  % 详解: 赋值：计算表达式并保存到 status
            else  % 详解: 条件判断：else 分支
                stepsize=stepsize/2;  % 详解: 赋值：计算表达式并保存到 stepsize
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        OPTIONS(18)=stepsize;  % 详解: 执行语句
        XOUT=OLDX;  % 详解: 赋值：计算表达式并保存到 XOUT
    elseif f<FIRSTF  % 详解: 条件判断：elseif (f<FIRSTF)
        [newstep,fbest] =cubici3(f,FIRSTF,gdnew,GDOLD,OPTIONS(18));  % 详解: 执行语句
        sk=(XOUT-OLDX)'*(GRAD-OLDG);  % 赋值：设置变量 sk  % 详解: 赋值：计算表达式并保存到 sk  % 详解: 赋值：计算表达式并保存到 sk
        if sk>1e-20  % 详解: 条件判断：if (sk>1e-20)
        how='';  % 详解: 赋值：计算表达式并保存到 how
            if gdnew<0  % 详解: 条件判断：if (gdnew<0)
                how='incstep';  % 详解: 赋值：计算表达式并保存到 how
                if newstep<OPTIONS(18),  newstep=2*OPTIONS(18)+1e-5; how=[how,' IF']; end  % 详解: 条件判断：if (newstep<OPTIONS(18),  newstep=2*OPTIONS(18)+1e-5; how=[how,' IF']; end)
                OPTIONS(18)=min([max([2,1.5*OPTIONS(18)]),1+sk+abs(gdnew)+max([0,OPTIONS(18)-1]), (1.2+0.3*(~OPTIONS(7)))*abs(newstep)]);  % 详解: 调用函数：OPTIONS(18)=min([max([2,1.5*OPTIONS(18)]),1+sk+abs(gdnew)+max([0,OPTIONS(18)-1]), (1.2+0.3*(~OPTIONS(7)))*abs(newstep)])
            else  % 详解: 条件判断：else 分支
                if OPTIONS(18)>0.9  % 详解: 条件判断：if (OPTIONS(18)>0.9)
                    how='int_st';  % 详解: 赋值：计算表达式并保存到 how
                    OPTIONS(18)=min([1,abs(newstep)]);  % 详解: 调用函数：OPTIONS(18)=min([1,abs(newstep)])
                end  % 详解: 执行语句
            end  % 详解: 执行语句
            [HESS,SD]=updhess(XOUT,OLDX,GRAD,OLDG,HESS,OPTIONS);  % 详解: 执行语句
            gdnew=GRAD'*SD;  % 赋值：设置变量 gdnew  % 详解: 赋值：计算表达式并保存到 gdnew  % 详解: 赋值：计算表达式并保存到 gdnew
            OLDX=XOUT;  % 详解: 赋值：计算表达式并保存到 OLDX
            status=4;  % 详解: 赋值：计算表达式并保存到 status
            FIRSTF=f;  % 详解: 赋值：计算表达式并保存到 FIRSTF
            OLDG=GRAD;  % 详解: 赋值：计算表达式并保存到 OLDG
            GDOLD=gdnew;  % 详解: 赋值：计算表达式并保存到 GDOLD
            if OPTIONS(7)==0, PCNT=1; MATX=zeros(3,1);  MATL(1)=f; end  % 详解: 条件判断：if (OPTIONS(7)==0, PCNT=1; MATX=zeros(3,1);  MATL(1)=f; end)
    elseif gdnew>0  % 详解: 条件判断：elseif (gdnew>0)
            how='inter_st';  % 详解: 赋值：计算表达式并保存到 how
            if OPTIONS(18)>0.01  % 详解: 条件判断：if (OPTIONS(18)>0.01)
                OPTIONS(18)=0.9*newstep;  % 详解: 执行语句
                XOUT=OLDX;  % 详解: 赋值：计算表达式并保存到 XOUT
            end  % 详解: 执行语句
            if OPTIONS(18)>1, OPTIONS(18)=1; end  % 详解: 条件判断：if (OPTIONS(18)>1, OPTIONS(18)=1; end)
        else  % 详解: 条件判断：else 分支
            OPTIONS(18)=max([min([newstep-OPTIONS(18),3]),0.5*OPTIONS(18)]);  % 详解: 调用函数：OPTIONS(18)=max([min([newstep-OPTIONS(18),3]),0.5*OPTIONS(18)])
            how='incst2';  % 详解: 赋值：计算表达式并保存到 how
            OLDX=XOUT;  % 详解: 赋值：计算表达式并保存到 OLDX
            FIRSTF=f;  % 详解: 赋值：计算表达式并保存到 FIRSTF
            OLDG=GRAD;  % 详解: 赋值：计算表达式并保存到 OLDG
            GDOLD=GRAD'*SD;  % 赋值：设置变量 GDOLD  % 详解: 赋值：计算表达式并保存到 GDOLD  % 详解: 赋值：计算表达式并保存到 GDOLD
                OLDX=XOUT;  % 详解: 赋值：计算表达式并保存到 OLDX
        end  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
        if gdnew<0&f>FIRSTF  % 详解: 条件判断：if (gdnew<0&f>FIRSTF)
            how='red_step';  % 详解: 赋值：计算表达式并保存到 how
            if norm(GRAD-OLDG)<1e-10; HESS=eye(nvars); end  % 详解: 条件判断：if (norm(GRAD-OLDG)<1e-10; HESS=eye(nvars); end)
            if abs(OPTIONS(18))<eps  % 详解: 条件判断：if (abs(OPTIONS(18))<eps)
                SD=norm(nvars,1)*(rand(nvars,1)-0.5);  % 详解: 赋值：将 norm(...) 的结果保存到 SD
                OPTIONS(18)=abs(rand(1,1)-0.5)*1e-6;  % 详解: 生成随机数/矩阵
                        how='RANDOM SD';  % 详解: 赋值：计算表达式并保存到 how
                    else  % 详解: 条件判断：else 分支
                        OPTIONS(18)=-OPTIONS(18)/2;  % 详解: 执行语句
            end  % 详解: 执行语句
            XOUT=OLDX;  % 详解: 赋值：计算表达式并保存到 XOUT
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    XOUT=XOUT+OPTIONS(18)*SD;  % 详解: 赋值：计算表达式并保存到 XOUT
    if isinf(OPTIONS(1))  % 详解: 条件判断：if (isinf(OPTIONS(1)))
       disp([num,how])  % 详解: 调用函数：disp([num,how])
    elseif OPTIONS(1)>0  % 详解: 条件判断：elseif (OPTIONS(1)>0)
       disp(num)  % 详解: 调用函数：disp(num)
    end  % 详解: 执行语句
end  % 详解: 执行语句

    if max(abs(SD))<2*OPTIONS(2) & (-GRAD'*SD) < 2*OPTIONS(3)  % 条件判断：if 分支开始  % 详解: 条件判断：if (max(abs(SD))<2*OPTIONS(2) & (-GRAD'*SD) < 2*OPTIONS(3))  % 详解: 条件判断：if (max(abs(SD))<2*OPTIONS(2) & (-GRAD'*SD) < 2*OPTIONS(3)  % 条件判断：if 分支开始  % 详解: 条件判断：if (max(abs(SD))<2*OPTIONS(2) & (-GRAD'*SD) < 2*OPTIONS(3)))
        if OPTIONS(1) > 0  % 详解: 条件判断：if (OPTIONS(1) > 0)
           disp('Optimization Terminated Successfully')  % 详解: 调用函数：disp('Optimization Terminated Successfully')
           disp(' Search direction less than 2*options(2)')  % 详解: 调用函数：disp(' Search direction less than 2*options(2)')
           disp(' Gradient in the search direction less than 2*options(3)')  % 详解: 调用函数：disp(' Gradient in the search direction less than 2*options(3)')
           disp([' NUMBER OF FUNCTION EVALUATIONS=', int2str(OPTIONS(10))]);  % 详解: 调用函数：disp([' NUMBER OF FUNCTION EVALUATIONS=', int2str(OPTIONS(10))])
        end  % 详解: 执行语句
        status=1;  % 详解: 赋值：计算表达式并保存到 status
    elseif OPTIONS(10)>OPTIONS(14)  % 详解: 条件判断：elseif (OPTIONS(10)>OPTIONS(14))
            if OPTIONS(1) >= 0  % 详解: 条件判断：if (OPTIONS(1) >= 0)
                  disp('Maximum number of function evaluations exceeded;')  % 详解: 调用函数：disp('Maximum number of function evaluations exceeded;')
                  disp('   increase options(14).')  % 详解: 调用函数：disp(' increase options(14).')
            end  % 详解: 执行语句
            status=1;  % 详解: 赋值：计算表达式并保存到 status
    else  % 详解: 条件判断：else 分支

        if PCNT~=0  % 详解: 条件判断：if (PCNT~=0)
            while PCNT > 0 & OPTIONS(10) <= OPTIONS(14)  % 详解: while 循环：当 (PCNT > 0 & OPTIONS(10) <= OPTIONS(14)) 为真时迭代
                x(:) = XOUT;  % 详解: 执行语句
                f = feval(funfcn,x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
                OPTIONS(10)=OPTIONS(10)+1;  % 详解: 执行语句
                [PCNT,MATL,MATX,steplen,f, how]=searchq(PCNT,f,OLDX,MATL,MATX,SD,GDOLD,OPTIONS(18), how);  % 详解: 执行语句
                OPTIONS(18)=steplen;  % 详解: 执行语句
                XOUT=OLDX+steplen*SD;  % 详解: 赋值：计算表达式并保存到 XOUT
            end  % 详解: 执行语句
            if OPTIONS(10)>OPTIONS(14)  % 详解: 条件判断：if (OPTIONS(10)>OPTIONS(14))
              if OPTIONS(1) >= 0  % 详解: 条件判断：if (OPTIONS(1) >= 0)
                  disp('Maximum number of function evaluations exceeded;')  % 详解: 调用函数：disp('Maximum number of function evaluations exceeded;')
                  disp('   increase options(14).')  % 详解: 调用函数：disp(' increase options(14).')
              end  % 详解: 执行语句
              status=1;  % 详解: 赋值：计算表达式并保存到 status
            end  % 详解: 执行语句
        else  % 详解: 条件判断：else 分支
            x(:)=XOUT;  % 详解: 执行语句
            f = feval(funfcn,x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
            OPTIONS(10)=OPTIONS(10)+1;  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

x(:)=XOUT;  % 详解: 执行语句
f = feval(funfcn,x,varargin{:});  % 详解: 赋值：将 feval(...) 的结果保存到 f
if f > FIRSTF  % 详解: 条件判断：if (f > FIRSTF)
    OPTIONS(8) = FIRSTF;  % 详解: 执行语句
    x(:)=OLDX;  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
    OPTIONS(8) = f;  % 详解: 执行语句
end  % 详解: 执行语句




