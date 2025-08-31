% 文件: LPINT.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [x,y]=lpint(f,G,h,lb,ub,x,n,id)  % 详解: 函数定义：lpint(f,G,h,lb,ub,x,n,id), 返回：x,y


global upper opt c N x0 A b ID;  % 详解: 执行语句
if nargin<8, id=ones(size(f));end  % 详解: 条件判断：if (nargin<8, id=ones(size(f));end)
if nargin<7|isempty(n), n=0;end  % 详解: 条件判断：if (nargin<7|isempty(n), n=0;end)
if nargin<6, x=[];end  % 详解: 条件判断：if (nargin<6, x=[];end)
if nargin<5|isempty(ub), ub=inf*ones(size(f));end  % 详解: 条件判断：if (nargin<5|isempty(ub), ub=inf*ones(size(f));end)
if nargin<4|isempty(lb), lb=zeros(size(f));end  % 详解: 条件判断：if (nargin<4|isempty(lb), lb=zeros(size(f));end)

upper=inf;  % 详解: 赋值：计算表达式并保存到 upper
c=f;N=n;x0=x;A=G;b=h;ID=id;  % 详解: 赋值：计算表达式并保存到 c
temp=ILP(lb(:),ub(:));  % 详解: 赋值：将 ILP(...) 的结果保存到 temp
x=opt;y=upper;  % 详解: 赋值：计算表达式并保存到 x

function y=ILP(vlb,vub)  % 详解: 执行语句
global upper opt c N x0 A b ID;  % 详解: 执行语句
  warning off;  % 详解: 执行语句
  [x,temp,how]=lp(c,A,b,vlb,vub,x0,N,-1);  % 详解: 执行语句
  if strcmp(how,'ok')~=1  % 详解: 条件判断：if (strcmp(how,'ok')~=1)
     return;  % 详解: 返回：从当前函数返回
  end;  % 详解: 执行语句
  if c'*x-upper>0.00005  %in order to avoid error  % 详解: 条件判断：if (c'*x-upper>0.00005)  % 详解: 条件判断：if (c'*x-upper>0.00005  %in order to avoid error  % 详解: 条件判断：if (c'*x-upper>0.00005))
     return;  % 详解: 返回：从当前函数返回
  end;  % 详解: 执行语句
   
  if max(abs(x.*ID-round(x.*ID)))<0.00005  % 详解: 条件判断：if (max(abs(x.*ID-round(x.*ID)))<0.00005)
    if upper-c'*x>0.00005 %in order to avoid error  % 详解: 条件判断：if (upper-c'*x>0.00005)  % 详解: 条件判断：if (upper-c'*x>0.00005 %in order to avoid error  % 详解: 条件判断：if (upper-c'*x>0.00005))
      opt=x';  % 赋值：设置变量 opt  % 详解: 赋值：计算表达式并保存到 opt  % 详解: 赋值：计算表达式并保存到 opt
      upper=c'*x;  % 赋值：设置变量 upper  % 详解: 赋值：计算表达式并保存到 upper  % 详解: 赋值：计算表达式并保存到 upper
      return;  % 详解: 返回：从当前函数返回
   else  % 详解: 条件判断：else 分支
      opt=[opt;x'];  % 赋值：设置变量 opt  % 详解: 赋值：计算表达式并保存到 opt  % 详解: 赋值：计算表达式并保存到 opt
      return;  % 详解: 返回：从当前函数返回
    end;  % 详解: 执行语句
  end;  % 详解: 执行语句
  notintx=find(abs(x-round(x))>=0.00005);  % 详解: 赋值：将 find(...) 的结果保存到 notintx
  intx=fix(x);  % 详解: 赋值：将 fix(...) 的结果保存到 intx
  tempvlb=vlb;  % 详解: 赋值：计算表达式并保存到 tempvlb
  tempvub=vub;  % 详解: 赋值：计算表达式并保存到 tempvub
  if vub(notintx(1,1),1)>=intx(notintx(1,1),1)+1  % 详解: 条件判断：if (vub(notintx(1,1),1)>=intx(notintx(1,1),1)+1)
     tempvlb(notintx(1,1),1)=intx(notintx(1,1),1)+1;  % 详解: 执行语句
     temp=ILP(tempvlb,vub);  % 详解: 赋值：将 ILP(...) 的结果保存到 temp
  end;  % 详解: 执行语句
  if vlb(notintx(1,1),1)<=intx(notintx(1,1),1)  % 详解: 条件判断：if (vlb(notintx(1,1),1)<=intx(notintx(1,1),1))
    tempvub(notintx(1,1),1)=intx(notintx(1,1),1);  % 详解: 调用函数：tempvub(notintx(1,1),1)=intx(notintx(1,1),1)
     temp=ILP(vlb,tempvub);  % 详解: 赋值：将 ILP(...) 的结果保存到 temp
  end;  % 详解: 执行语句



