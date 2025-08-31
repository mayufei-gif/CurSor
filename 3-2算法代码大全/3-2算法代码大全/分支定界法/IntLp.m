% 文件: IntLp.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x,y]=IntLp(f,G,h,Geq,heq,lb,ub,x,id,options)  % 详解: 函数定义：IntLp(f,G,h,Geq,heq,lb,ub,x,id,options), 返回：x,y



global upper opt c x0 A b Aeq beq ID options;  % 详解: 执行语句
if nargin<10, options=optimset({});options.Display='off';  % 详解: 条件判断：if (nargin<10, options=optimset({});options.Display='off';)
options.LargeScale='off';end  % 详解: 赋值：计算表达式并保存到 options.LargeScale
if nargin<9, id=ones(size(f));end  % 详解: 条件判断：if (nargin<9, id=ones(size(f));end)
if nargin<8, x=[];end  % 详解: 条件判断：if (nargin<8, x=[];end)
if nargin<7|isempty(ub), ub=inf*ones(size(f));end  % 详解: 条件判断：if (nargin<7|isempty(ub), ub=inf*ones(size(f));end)
if nargin<6|isempty(lb), lb=zeros(size(f));end  % 详解: 条件判断：if (nargin<6|isempty(lb), lb=zeros(size(f));end)
if nargin<5, heq=[];end  % 详解: 条件判断：if (nargin<5, heq=[];end)
if nargin<4, Geq=[];end  % 详解: 条件判断：if (nargin<4, Geq=[];end)
upper=inf;c=f;x0=x;A=G;b=h;Aeq=Geq;beq=heq;ID=id;  % 详解: 赋值：计算表达式并保存到 upper
ftemp=ILP(lb(:),ub(:));  % 详解: 赋值：将 ILP(...) 的结果保存到 ftemp
x=opt;y=upper;  % 详解: 赋值：计算表达式并保存到 x
function ftemp=ILP(vlb,vub)  % 详解: 执行语句
global upper opt c x0 A b Aeq beq ID options;  % 详解: 执行语句
[x,ftemp,how]=linprog(c,A,b,Aeq,beq,vlb,vub,x0,options);  % 详解: 执行语句
if how<=0  % 详解: 条件判断：if (how<=0)
return;  % 详解: 返回：从当前函数返回
end;  % 详解: 执行语句
if ftemp-upper>0.00005  % 详解: 条件判断：if (ftemp-upper>0.00005)
return;  % 详解: 返回：从当前函数返回
end;  % 详解: 执行语句
if max(abs(x.*ID-round(x.*ID)))<0.00005  % 详解: 条件判断：if (max(abs(x.*ID-round(x.*ID)))<0.00005)
if upper-ftemp>0.00005  % 详解: 条件判断：if (upper-ftemp>0.00005)
opt=x';upper=ftemp;  % 赋值：设置变量 opt  % 详解: 赋值：计算表达式并保存到 opt  % 详解: 赋值：计算表达式并保存到 opt
return;  % 详解: 返回：从当前函数返回
else  % 详解: 条件判断：else 分支
opt=[opt;x'];  % 赋值：设置变量 opt  % 详解: 赋值：计算表达式并保存到 opt  % 详解: 赋值：计算表达式并保存到 opt
return;  % 详解: 返回：从当前函数返回
end;end;  % 详解: 执行语句
notintx=find(abs(x-round(x))>=0.00005);  % 详解: 赋值：将 find(...) 的结果保存到 notintx
intx=fix(x);tempvlb=vlb;tempvub=vub;  % 详解: 赋值：将 fix(...) 的结果保存到 intx
if vub(notintx(1,1),1)>=intx(notintx(1,1),1)+1  % 详解: 条件判断：if (vub(notintx(1,1),1)>=intx(notintx(1,1),1)+1)
tempvlb(notintx(1,1),1)=intx(notintx(1,1),1)+1;  % 详解: 执行语句
ftemp=ILP(tempvlb,vub);  % 详解: 赋值：将 ILP(...) 的结果保存到 ftemp
end;  % 详解: 执行语句
if vlb(notintx(1,1),1)<=intx(notintx(1,1),1)  % 详解: 条件判断：if (vlb(notintx(1,1),1)<=intx(notintx(1,1),1))
tempvub(notintx(1,1),1)=intx(notintx(1,1),1);  % 详解: 调用函数：tempvub(notintx(1,1),1)=intx(notintx(1,1),1)
ftemp=ILP(vlb,tempvub);  % 详解: 赋值：将 ILP(...) 的结果保存到 ftemp
end;  % 详解: 执行语句




