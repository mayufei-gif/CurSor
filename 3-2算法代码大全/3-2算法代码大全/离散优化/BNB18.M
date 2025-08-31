% 文件: BNB18.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [errmsg,Z,X,t,c,fail] = BNB18(fun,x0,xstat,xl,xu,A,B,Aeq,Beq,nonlcon,setts,options1,options2,maxSQPit,varargin);  % 详解: 函数定义：BNB18(fun,x0,xstat,xl,xu,A,B,Aeq,Beq,nonlcon,setts,options1,options2,maxSQPit,varargin), 返回：errmsg,Z,X,t,c,fail


global maxSQPiter;  % 详解: 执行语句

Z=[]; X=[]; t=0; c=0; fail=0;  % 详解: 赋值：计算表达式并保存到 Z
if nargin<2, errmsg='BNB needs at least 2 input arguments.'; return; end;  % 详解: 条件判断：if (nargin<2, errmsg='BNB needs at least 2 input arguments.'; return; end;)
if isempty(fun), errmsg='No fun found.'; return; end;  % 详解: 条件判断：if (isempty(fun), errmsg='No fun found.'; return; end;)
if isempty(x0), errmsg='No x0 found.'; return;  % 详解: 条件判断：if (isempty(x0), errmsg='No x0 found.'; return;)
elseif size(x0,2)>1, errmsg='x0 must be a column vector.'; return; end;  % 详解: 条件判断：elseif (size(x0,2)>1, errmsg='x0 must be a column vector.'; return; end;)
xstatus=zeros(size(x0));  % 详解: 赋值：将 zeros(...) 的结果保存到 xstatus
if nargin>2 & ~isempty(xstat)  % 详解: 条件判断：if (nargin>2 & ~isempty(xstat))
   if all(size(xstat)<=size(x0))  % 详解: 条件判断：if (all(size(xstat)<=size(x0)))
      xstatus(1:size(xstat))=xstat;  % 详解: 获取向量/矩阵尺寸
   else errmsg='xstatus must be a column vector the same size as x0.'; return;  % 详解: 条件判断：else 分支
   end;  % 详解: 执行语句
   if any(xstatus~=round(xstatus) | xstatus<0 | 2<xstatus)  % 详解: 条件判断：if (any(xstatus~=round(xstatus) | xstatus<0 | 2<xstatus))
      errmsg='xstatus must consist of the integers 0,1 en 2.'; return;  % 详解: 赋值：计算表达式并保存到 errmsg
   end;  % 详解: 执行语句
end;  % 详解: 执行语句
xlb=zeros(size(x0));  % 详解: 赋值：将 zeros(...) 的结果保存到 xlb
xlb(find(xstatus==0))=-inf;  % 详解: 执行语句
if nargin>3 & ~isempty(xl)  % 详解: 条件判断：if (nargin>3 & ~isempty(xl))
   if all(size(xl)<=size(x0))  % 详解: 条件判断：if (all(size(xl)<=size(x0)))
      xlb(1:size(xl,1))=xl;  % 详解: 获取向量/矩阵尺寸
   else errmsg='xlb must be a column vector the same size as x0.'; return;  % 详解: 条件判断：else 分支
   end;  % 详解: 执行语句
end;  % 详解: 执行语句
if any(x0<xlb)  % 详解: 条件判断：if (any(x0<xlb))
   errmsg='x0 must be in the range xlb <= x0.'; return;  % 详解: 赋值：计算表达式并保存到 errmsg
elseif any(xstatus==1 & (~isfinite(xlb) | xlb~=round(xlb)))  % 详解: 条件判断：elseif (any(xstatus==1 & (~isfinite(xlb) | xlb~=round(xlb))))
   errmsg='xlb(i) must be an integer if x(i) is an integer variabele.'; return;  % 详解: 赋值：计算表达式并保存到 errmsg
end;  % 详解: 执行语句
xlb(find(xstatus==2))=x0(find(xstatus==2));  % 详解: 调用函数：xlb(find(xstatus==2))=x0(find(xstatus==2))
xub=ones(size(x0));  % 详解: 赋值：将 ones(...) 的结果保存到 xub
xub(find(xstatus==0))=inf;  % 详解: 执行语句
if nargin>4 & ~isempty(xu)  % 详解: 条件判断：if (nargin>4 & ~isempty(xu))
   if all(size(xu)<=size(x0))  % 详解: 条件判断：if (all(size(xu)<=size(x0)))
      xub(1:size(xu,1))=xu;  % 详解: 获取向量/矩阵尺寸
   else errmsg='xub must be a column vector the same size as x0.'; return;  % 详解: 条件判断：else 分支
   end;  % 详解: 执行语句
end;  % 详解: 执行语句
if any(x0>xub)  % 详解: 条件判断：if (any(x0>xub))
   errmsg='x0 must be in the range x0 <=xub.'; return;  % 详解: 赋值：计算表达式并保存到 errmsg
elseif any(xstatus==1 & (~isfinite(xub) | xub~=round(xub)))  % 详解: 条件判断：elseif (any(xstatus==1 & (~isfinite(xub) | xub~=round(xub))))
   errmsg='xub(i) must be an integer if x(i) is an integer variabale.'; return;  % 详解: 赋值：计算表达式并保存到 errmsg
end;  % 详解: 执行语句
xub(find(xstatus==2))=x0(find(xstatus==2));  % 详解: 调用函数：xub(find(xstatus==2))=x0(find(xstatus==2))
if nargin>5  % 详解: 条件判断：if (nargin>5)
   if ~isempty(A) & size(A,2)~=size(x0,1), errmsg='Matrix A not correct.'; return; end;  % 详解: 条件判断：if (~isempty(A) & size(A,2)~=size(x0,1), errmsg='Matrix A not correct.'; return; end;)
else A=[]; end;  % 详解: 条件判断：else 分支
if nargin>6  % 详解: 条件判断：if (nargin>6)
   if ~isempty(B) & any(size(B)~=[size(A,1) 1]), errmsg='Column vector B not correct.'; return; end;  % 详解: 条件判断：if (~isempty(B) & any(size(B)~=[size(A,1) 1]), errmsg='Column vector B not correct.'; return; end;)
else B=[]; end;  % 详解: 条件判断：else 分支
if isempty(A) & ~isempty(B), errmsg='A and B should only be nonempty together.'; return; end;  % 详解: 条件判断：if (isempty(A) & ~isempty(B), errmsg='A and B should only be nonempty together.'; return; end;)
if isempty(B) & ~isempty(A), B=zeros(size(A,1),1); end;  % 详解: 条件判断：if (isempty(B) & ~isempty(A), B=zeros(size(A,1),1); end;)
if nargin>7 & ~isempty(Aeq)  % 详解: 条件判断：if (nargin>7 & ~isempty(Aeq))
   if size(Aeq,2)~=size(x0,1), errmsg='Matrix Aeq not correct.'; return; end;  % 详解: 条件判断：if (size(Aeq,2)~=size(x0,1), errmsg='Matrix Aeq not correct.'; return; end;)
else Aeq=[]; end;  % 详解: 条件判断：else 分支
if nargin>8  % 详解: 条件判断：if (nargin>8)
   if ~isempty(Beq) & any(size(Beq)~=[size(Aeq,1) 1]), errmsg='Column vector Beq not correct.'; return; end;  % 详解: 条件判断：if (~isempty(Beq) & any(size(Beq)~=[size(Aeq,1) 1]), errmsg='Column vector Beq not correct.'; return; end;)
else Beq=[]; end;  % 详解: 条件判断：else 分支
if isempty(Aeq) & ~isempty(Beq), errmsg='Aeq and Beq should only be nonempty together'; return; end;  % 详解: 条件判断：if (isempty(Aeq) & ~isempty(Beq), errmsg='Aeq and Beq should only be nonempty together'; return; end;)
if isempty(Beq) & ~isempty(Aeq), Beq=zeros(size(Aeq,1),1); end;  % 详解: 条件判断：if (isempty(Beq) & ~isempty(Aeq), Beq=zeros(size(Aeq,1),1); end;)
if nargin<10, nonlcon=''; end;  % 详解: 条件判断：if (nargin<10, nonlcon=''; end;)
settings = [0 0 0];  % 详解: 赋值：计算表达式并保存到 settings
if nargin>10 & ~isempty(setts)  % 详解: 条件判断：if (nargin>10 & ~isempty(setts))
   if all(size(setts)<=size(settings))  % 详解: 条件判断：if (all(size(setts)<=size(settings)))
      settings(setts~=0)=setts(setts~=0);  % 详解: 调用函数：settings(setts~=0)=setts(setts~=0)
   else errmsg='settings should be a row vector of length 3.'; return; end;  % 详解: 条件判断：else 分支
end;  % 详解: 执行语句
if nargin<12, options1=[]; end;  % 详解: 条件判断：if (nargin<12, options1=[]; end;)
options1=optimset(optimset('fmincon'),options1);  % 详解: 赋值：将 optimset(...) 的结果保存到 options1
if nargin<13, options2=[]; end;  % 详解: 条件判断：if (nargin<13, options2=[]; end;)
options2=optimset(optimset('fmincon'),options2);  % 详解: 赋值：将 optimset(...) 的结果保存到 options2
if nargin<14, maxSQPiter=1000;  % 详解: 条件判断：if (nargin<14, maxSQPiter=1000;)
elseif isnumeric(maxSQPit) & all(size(maxSQPit))==1 & maxSQPit>0 & round(maxSQPit)==maxSQPit  % 详解: 条件判断：elseif (isnumeric(maxSQPit) & all(size(maxSQPit))==1 & maxSQPit>0 & round(maxSQPit)==maxSQPit)
   maxSQPiter=maxSQPit;  % 详解: 赋值：计算表达式并保存到 maxSQPiter
else errmsg='maxSQPiter must be an integer >0'; return; end;  % 详解: 条件判断：else 分支
eval(['z=',fun,'(x0,varargin{:});'],'errmsg=''fun caused error.''; return;');  % 详解: 调用函数：eval(['z=',fun,'(x0,varargin{:});'],'errmsg=''fun caused error.''; return;')
if ~isempty(nonlcon)  % 详解: 条件判断：if (~isempty(nonlcon))
   eval(['[C, Ceq]=',nonlcon,'(x0,varargin{:});'],'errmsg=''nonlcon caused error.''; return;');  % 详解: 调用函数：eval(['[C, Ceq]=',nonlcon,'(x0,varargin{:});'],'errmsg=''nonlcon caused error.''; return;')
   if size(C,2)>1 | size(Ceq,2)>1, errmsg='C en Ceq must be column vectors.'; return; end;  % 详解: 条件判断：if (size(C,2)>1 | size(Ceq,2)>1, errmsg='C en Ceq must be column vectors.'; return; end;)
end;  % 详解: 执行语句

currentwarningstate=warning;  % 详解: 赋值：计算表达式并保存到 currentwarningstate
warning off;  % 详解: 执行语句
tic;  % 详解: 执行语句
lx = size(x0,1);  % 详解: 赋值：将 size(...) 的结果保存到 lx
z_incumbent=inf;  % 详解: 赋值：计算表达式并保存到 z_incumbent
x_incumbent=inf*ones(size(x0));  % 详解: 赋值：计算表达式并保存到 x_incumbent
I = ceil(sum(log2(xub(find(xstatus==1))-xlb(find(xstatus==1))+1))+size(find(xstatus==1),1)+1);  % 详解: 赋值：将 ceil(...) 的结果保存到 I
stackx0=zeros(lx,I);  % 详解: 赋值：将 zeros(...) 的结果保存到 stackx0
stackx0(:,1)=x0;  % 详解: 执行语句
stackxlb=zeros(lx,I);  % 详解: 赋值：将 zeros(...) 的结果保存到 stackxlb
stackxlb(:,1)=xlb;  % 详解: 执行语句
stackxub=zeros(lx,I);  % 详解: 赋值：将 zeros(...) 的结果保存到 stackxub
stackxub(:,1)=xub;  % 详解: 执行语句
stacksize=1;  % 详解: 赋值：计算表达式并保存到 stacksize
xchoice=zeros(size(x0));  % 详解: 赋值：将 zeros(...) 的结果保存到 xchoice
if ~isempty(Aeq)  % 详解: 条件判断：if (~isempty(Aeq))
   j=0;  % 详解: 赋值：计算表达式并保存到 j
   for i=1:size(Aeq,1)  % 详解: for 循环：迭代变量 i 遍历 1:size(Aeq,1)
      if Beq(i)==1 & all(Aeq(i,:)==0 | Aeq(i,:)==1)  % 详解: 条件判断：if (Beq(i)==1 & all(Aeq(i,:)==0 | Aeq(i,:)==1))
         J=find(Aeq(i,:)==1);  % 详解: 赋值：将 find(...) 的结果保存到 J
         if all(xstatus(J)~=0 & xchoice(J)==0 & xlb(J)==0 & xub(J)==1)  % 详解: 条件判断：if (all(xstatus(J)~=0 & xchoice(J)==0 & xlb(J)==0 & xub(J)==1))
            if all(xstatus(J)~=2) | all(x0(J(find(xstatus(J)==2)))==0)  % 详解: 条件判断：if (all(xstatus(J)~=2) | all(x0(J(find(xstatus(J)==2)))==0))
               j=j+1;  % 详解: 赋值：计算表达式并保存到 j
               xchoice(J)=j;  % 详解: 执行语句
               if sum(x0(J))==0, errmsg='x0 not correct.'; return; end;  % 详解: 条件判断：if (sum(x0(J))==0, errmsg='x0 not correct.'; return; end;)
            end;  % 详解: 执行语句
         end;  % 详解: 执行语句
      end;  % 详解: 执行语句
   end;  % 详解: 执行语句
end;  % 详解: 执行语句
errx=optimget(options2,'TolX');  % 详解: 赋值：将 optimget(...) 的结果保存到 errx
errcon=optimget(options2,'TolCon');  % 详解: 赋值：将 optimget(...) 的结果保存到 errcon
fail=0;  % 详解: 赋值：计算表达式并保存到 fail
c=0;  % 详解: 赋值：计算表达式并保存到 c

while stacksize>0  % 详解: while 循环：当 (stacksize>0) 为真时迭代
   c=c+1;  % 详解: 赋值：计算表达式并保存到 c
   
   x0=stackx0(:,stacksize);  % 详解: 赋值：将 stackx0(...) 的结果保存到 x0
   xlb=stackxlb(:,stacksize);  % 详解: 赋值：将 stackxlb(...) 的结果保存到 xlb
   xub=stackxub(:,stacksize);  % 详解: 赋值：将 stackxub(...) 的结果保存到 xub
   x0(find(x0<xlb))=xlb(find(x0<xlb));  % 详解: 调用函数：x0(find(x0<xlb))=xlb(find(x0<xlb))
   x0(find(x0>xub))=xub(find(x0>xub));  % 详解: 调用函数：x0(find(x0>xub))=xub(find(x0>xub))
   stacksize=stacksize-1;  % 详解: 赋值：计算表达式并保存到 stacksize
      
   
   con=BNBCON(x0,A,B,Aeq,Beq,xlb,xub,nonlcon,varargin{:});  % 详解: 赋值：将 BNBCON(...) 的结果保存到 con
   if abs(con)>errcon & settings(1)~=0  % 详解: 条件判断：if (abs(con)>errcon & settings(1)~=0)
      [x1 dummy feasflag]=fmincon('0',x0,A,B,Aeq,Beq,xlb,xub,nonlcon,options1,varargin{:});  % 详解: 执行语句
      if settings(3) & feasflag==0  % 详解: 条件判断：if (settings(3) & feasflag==0)
         con=BNBCON(x1,A,B,Aeq,Beq,xlb,xub,nonlcon,varargin{:});  % 详解: 赋值：将 BNBCON(...) 的结果保存到 con
         if con<errcon, feasflag=1; end;  % 详解: 条件判断：if (con<errcon, feasflag=1; end;)
      end;  % 详解: 执行语句
   else x1=x0; feasflag=1; end;  % 详解: 条件判断：else 分支
   
   if feasflag>0  % 详解: 条件判断：if (feasflag>0)
      [x z convflag]=fmincon(fun,x1,A,B,Aeq,Beq,xlb,xub,nonlcon,options2,varargin{:});  % 详解: 执行语句
      if settings(3) & convflag==0  % 详解: 条件判断：if (settings(3) & convflag==0)
         con=BNBCON(x,A,B,Aeq,Beq,xlb,xub,nonlcon,varargin{:});  % 详解: 赋值：将 BNBCON(...) 的结果保存到 con
         if con<errcon, convflag=1; end;  % 详解: 条件判断：if (con<errcon, convflag=1; end;)
      end;  % 详解: 执行语句
   else convflag=feasflag; end;  % 详解: 条件判断：else 分支
   
   K = find(xstatus==1 & xlb~=xub);  % 详解: 赋值：将 find(...) 的结果保存到 K
   separation=1;  % 详解: 赋值：计算表达式并保存到 separation
   if convflag<0 | (convflag==0 & settings(2))  % 详解: 条件判断：if (convflag<0 | (convflag==0 & settings(2)))
      separation=0;  % 详解: 赋值：计算表达式并保存到 separation
   elseif z>=z_incumbent & convflag>0  % 详解: 条件判断：elseif (z>=z_incumbent & convflag>0)
      separation=0;  % 详解: 赋值：计算表达式并保存到 separation
   elseif all(abs(round(x(K))-x(K))<errx) & convflag>0  % 详解: 条件判断：elseif (all(abs(round(x(K))-x(K))<errx) & convflag>0)
      z_incumbent = z;  % 详解: 赋值：计算表达式并保存到 z_incumbent
      x_incumbent = x;  % 详解: 赋值：计算表达式并保存到 x_incumbent
      separation = 0;  % 详解: 赋值：计算表达式并保存到 separation
	end;  % 详解: 执行语句
   
   if separation == 1 & ~isempty(K)  % 详解: 条件判断：if (separation == 1 & ~isempty(K))
      dzsep=-1;  % 详解: 赋值：计算表达式并保存到 dzsep
      for i=1:size(K,1)  % 详解: for 循环：迭代变量 i 遍历 1:size(K,1)
         dxsepc = abs(round(x(K(i)))-x(K(i)));  % 详解: 赋值：将 abs(...) 的结果保存到 dxsepc
         if dxsepc>=errx | convflag==0  % 详解: 条件判断：if (dxsepc>=errx | convflag==0)
            xsepc = x; xsepc(K(i))=round(x(K(i)));  % 详解: 赋值：计算表达式并保存到 xsepc
            dzsepc = abs(feval(fun,xsepc,varargin{:})-z);  % 详解: 赋值：将 abs(...) 的结果保存到 dzsepc
            if dzsepc>dzsep  % 详解: 条件判断：if (dzsepc>dzsep)
               dzsep=dzsepc;  % 详解: 赋值：计算表达式并保存到 dzsep
               ixsep=K(i);  % 详解: 赋值：将 K(...) 的结果保存到 ixsep
            end;  % 详解: 执行语句
         end;  % 详解: 执行语句
      end;  % 详解: 执行语句
      
      if xchoice(ixsep)==0  % 详解: 条件判断：if (xchoice(ixsep)==0)
         
         branch=1;  % 详解: 赋值：计算表达式并保存到 branch
         domain=[xlb(ixsep) xub(ixsep)];  % 详解: 赋值：计算表达式并保存到 domain
         while branch==1  % 详解: while 循环：当 (branch==1) 为真时迭代
            xboundary=(domain(1)+domain(2))/2;  % 详解: 赋值：计算表达式并保存到 xboundary
            if x(ixsep)<xboundary  % 详解: 条件判断：if (x(ixsep)<xboundary)
               domainA=[domain(1) floor(xboundary)];  % 详解: 赋值：计算表达式并保存到 domainA
               domainB=[floor(xboundary+1) domain(2)];  % 详解: 赋值：计算表达式并保存到 domainB
            else  % 详解: 条件判断：else 分支
               domainA=[floor(xboundary+1) domain(2)];  % 详解: 赋值：计算表达式并保存到 domainA
               domainB=[domain(1) floor(xboundary)];  % 详解: 赋值：计算表达式并保存到 domainB
            end;  % 详解: 执行语句
            stacksize=stacksize+1;  % 详解: 赋值：计算表达式并保存到 stacksize
            stackx0(:,stacksize)=x;  % 详解: 执行语句
            stackxlb(:,stacksize)=xlb;  % 详解: 执行语句
            stackxlb(ixsep,stacksize)=domainB(1);  % 详解: 调用函数：stackxlb(ixsep,stacksize)=domainB(1)
            stackxub(:,stacksize)=xub;  % 详解: 执行语句
            stackxub(ixsep,stacksize)=domainB(2);  % 详解: 调用函数：stackxub(ixsep,stacksize)=domainB(2)
            if domainA(1)==domainA(2)  % 详解: 条件判断：if (domainA(1)==domainA(2))
               stacksize=stacksize+1;  % 详解: 赋值：计算表达式并保存到 stacksize
               stackx0(:,stacksize)=x;  % 详解: 执行语句
               stackxlb(:,stacksize)=xlb;  % 详解: 执行语句
               stackxlb(ixsep,stacksize)=domainA(1);  % 详解: 调用函数：stackxlb(ixsep,stacksize)=domainA(1)
               stackxub(:,stacksize)=xub;  % 详解: 执行语句
               stackxub(ixsep,stacksize)=domainA(2);  % 详解: 调用函数：stackxub(ixsep,stacksize)=domainA(2)
               branch=0;  % 详解: 赋值：计算表达式并保存到 branch
            else  % 详解: 条件判断：else 分支
               domain=domainA;  % 详解: 赋值：计算表达式并保存到 domain
               branch=1;  % 详解: 赋值：计算表达式并保存到 branch
            end;  % 详解: 执行语句
         end;  % 详解: 执行语句
      else  % 详解: 条件判断：else 分支
         
         L=find(xchoice==xchoice(ixsep));  % 详解: 赋值：将 find(...) 的结果保存到 L
         M=intersect(K,L);  % 详解: 赋值：将 intersect(...) 的结果保存到 M
         [dummy,N]=sort(x(M));  % 详解: 执行语句
         part1=M(N(1:floor(size(N)/2))); part2=M(N(floor(size(N)/2)+1:size(N)));  % 详解: 赋值：将 M(...) 的结果保存到 part1
         stacksize=stacksize+1;  % 详解: 赋值：计算表达式并保存到 stacksize
         stackx0(:,stacksize)=x;  % 详解: 执行语句
         O = (1-sum(stackx0(part1,stacksize)))/size(part1,1);  % 详解: 赋值：计算表达式并保存到 O
         stackx0(part1,stacksize)=stackx0(part1,stacksize)+O;  % 详解: 执行语句
         stackxlb(:,stacksize)=xlb;  % 详解: 执行语句
         stackxub(:,stacksize)=xub;  % 详解: 执行语句
         stackxub(part2,stacksize)=0;  % 详解: 执行语句
         stacksize=stacksize+1;  % 详解: 赋值：计算表达式并保存到 stacksize
         stackx0(:,stacksize)=x;  % 详解: 执行语句
         O = (1-sum(stackx0(part2,stacksize)))/size(part2,1);  % 详解: 赋值：计算表达式并保存到 O
         stackx0(part2,stacksize)=stackx0(part2,stacksize)+O;  % 详解: 执行语句
         stackxlb(:,stacksize)=xlb;  % 详解: 执行语句
         stackxub(:,stacksize)=xub;  % 详解: 执行语句
         stackxub(part1,stacksize)=0;  % 详解: 执行语句
         if size(part2,1)==1, stackxlb(part2,stacksize)=1; end;  % 详解: 条件判断：if (size(part2,1)==1, stackxlb(part2,stacksize)=1; end;)
      end;  % 详解: 执行语句
   elseif separation==1 & isempty(K)  % 详解: 条件判断：elseif (separation==1 & isempty(K))
      fail=fail+1;  % 详解: 赋值：计算表达式并保存到 fail
   end;  % 详解: 执行语句
end;  % 详解: 执行语句

t=toc;  % 详解: 赋值：计算表达式并保存到 t
Z = z_incumbent;  % 详解: 赋值：计算表达式并保存到 Z
X = x_incumbent;  % 详解: 赋值：计算表达式并保存到 X
errmsg='';  % 详解: 赋值：计算表达式并保存到 errmsg
eval(['warning ',currentwarningstate]);  % 详解: 调用函数：eval(['warning ',currentwarningstate])

function CON=BNBCON(x,A,B,Aeq,Beq,xlb,xub,nonlcon,varargin);  % 详解: 执行语句
if isempty(A), CON1=[]; else CON1 = max(A*x-B,0); end;  % 详解: 条件判断：if (isempty(A), CON1=[]; else CON1 = max(A*x-B,0); end;)
if isempty(Aeq), CON2=[]; else CON2 = abs(Aeq*x-Beq); end;  % 详解: 条件判断：if (isempty(Aeq), CON2=[]; else CON2 = abs(Aeq*x-Beq); end;)
CON3 = max(xlb-x,0);  % 详解: 赋值：将 max(...) 的结果保存到 CON3
CON4 = max(x-xub,0);  % 详解: 赋值：将 max(...) 的结果保存到 CON4
if isempty(nonlcon)  % 详解: 条件判断：if (isempty(nonlcon))
   CON5=[]; CON6=[];  % 详解: 赋值：计算表达式并保存到 CON5
else  % 详解: 条件判断：else 分支
   [C Ceq]=feval(nonlcon,x,varargin{:});  % 详解: 执行语句
   CON5 = max(C,0);  % 详解: 赋值：将 max(...) 的结果保存到 CON5
   CON6 = abs(Ceq);  % 详解: 赋值：将 abs(...) 的结果保存到 CON6
end;  % 详解: 执行语句
CON  = max([CON1; CON2; CON3; CON4; CON5; CON6]);  % 详解: 赋值：将 max(...) 的结果保存到 CON



