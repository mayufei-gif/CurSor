% 文件: DYNPROG.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [p_opt,fval]=dynprog(x,DecisFun,ObjFun,TransFun)  % 详解: 函数定义：dynprog(x,DecisFun,ObjFun,TransFun), 返回：p_opt,fval


k=length(x(1,:));f_opt=nan*ones(size(x));d_opt=f_opt;  % 详解: 赋值：将 length(...) 的结果保存到 k
t_vubm=inf*ones(size(x));x_isnan=~isnan(x);t_vub=inf;  % 详解: 赋值：计算表达式并保存到 t_vubm
tmp1=find(x_isnan(:,k));tmp2=length(tmp1);  % 详解: 赋值：将 find(...) 的结果保存到 tmp1
for i=1:tmp2  % 详解: for 循环：迭代变量 i 遍历 1:tmp2
   u=feval(DecisFun,k,x(i,k));tmp3=length(u);  % 详解: 赋值：将 feval(...) 的结果保存到 u
   for j=1:tmp3  % 详解: for 循环：迭代变量 j 遍历 1:tmp3
         tmp=feval(ObjFun,k,x(tmp1(i),k),u(j));  % 详解: 赋值：将 feval(...) 的结果保存到 tmp
         if tmp<=t_vub,  % 详解: 条件判断：if (tmp<=t_vub,)
            f_opt(i,k)=tmp;d_opt(i,k)=u(j);t_vub=tmp;  % 详解: 执行语句
end;end;end  % 详解: 执行语句
for ii=k-1:-1:1  % 详解: for 循环：迭代变量 ii 遍历 k-1:-1:1
   tmp10=find(x_isnan(:,ii));tmp20=length(tmp10);  % 详解: 赋值：将 find(...) 的结果保存到 tmp10
   for i=1:tmp20  % 详解: for 循环：迭代变量 i 遍历 1:tmp20
      u=feval(DecisFun,ii,x(i,ii));tmp30=length(u);  % 详解: 赋值：将 feval(...) 的结果保存到 u
      for j=1:tmp30  % 详解: for 循环：迭代变量 j 遍历 1:tmp30
         tmp00=feval(ObjFun,ii,x(tmp10(i),ii),u(j));  % 详解: 赋值：将 feval(...) 的结果保存到 tmp00
         tmp40=feval(TransFun,ii,x(tmp10(i),ii),u(j));  % 详解: 赋值：将 feval(...) 的结果保存到 tmp40
         tmp50=x(:,ii+1)-tmp40;  % 详解: 赋值：将 x(...) 的结果保存到 tmp50
         tmp60=find(tmp50==0);  % 详解: 赋值：将 find(...) 的结果保存到 tmp60
         if ~isempty(tmp60),  % 详解: 条件判断：if (~isempty(tmp60),)
            tmp00=tmp00+f_opt(tmp60(1),ii+1);  % 详解: 赋值：计算表达式并保存到 tmp00
            if tmp00<=t_vubm(i,ii)  % 详解: 条件判断：if (tmp00<=t_vubm(i,ii))
               f_opt(i,ii)=tmp00;d_opt(i,ii)=u(j);  % 详解: 调用函数：f_opt(i,ii)=tmp00;d_opt(i,ii)=u(j)
               t_vubm(i,ii)=tmp00;  % 详解: 执行语句
end;end;end;end;end;  % 详解: 执行语句
fval=f_opt(tmp1,1);  % 详解: 赋值：将 f_opt(...) 的结果保存到 fval
p_opt=[];tmpx=[];tmpd=[];tmpf=[];  % 详解: 赋值：计算表达式并保存到 p_opt
tmp0=find(x_isnan(:,1));tmp01=length(tmp0);  % 详解: 赋值：将 find(...) 的结果保存到 tmp0
for i=1:tmp01,  % 详解: for 循环：迭代变量 i 遍历 1:tmp01,
  tmpd(i)=d_opt(tmp0(i),1);  % 详解: 调用函数：tmpd(i)=d_opt(tmp0(i),1)
  tmpx(i)=x(tmp0(i),1);  % 详解: 调用函数：tmpx(i)=x(tmp0(i),1)
  tmpf(i)=feval(ObjFun,1,tmpx(i),tmpd(i));  % 详解: 调用函数：tmpf(i)=feval(ObjFun,1,tmpx(i),tmpd(i))
  p_opt(k*(i-1)+1,[1,2,3,4])=[1,tmpx(i),...  % 详解: 执行语句
tmpd(i),tmpf(i)];  % 详解: 执行语句
  for ii=2:k  % 详解: for 循环：迭代变量 ii 遍历 2:k
     tmpx(i)=feval(TransFun,ii-1,tmpx(i),tmpd(i));  % 详解: 调用函数：tmpx(i)=feval(TransFun,ii-1,tmpx(i),tmpd(i))
     tmp1=x(:,ii)-tmpx(i);tmp2=find(tmp1==0);  % 详解: 赋值：将 x(...) 的结果保存到 tmp1
     if ~isempty(tmp2)  % 详解: 条件判断：if (~isempty(tmp2))
        tmpd(i)=d_opt(tmp2(1),ii);  % 详解: 调用函数：tmpd(i)=d_opt(tmp2(1),ii)
     end;  % 详解: 执行语句
     tmpf(i)=feval(ObjFun,ii,tmpx(i),tmpd(i));  % 详解: 调用函数：tmpf(i)=feval(ObjFun,ii,tmpx(i),tmpd(i))
     p_opt(k*(i-1)+ii,[1,2,3,4])=[ii,tmpx(i),...  % 详解: 执行语句
tmpd(i),tmpf(i)];  % 详解: 执行语句
end;end;  % 详解: 执行语句





