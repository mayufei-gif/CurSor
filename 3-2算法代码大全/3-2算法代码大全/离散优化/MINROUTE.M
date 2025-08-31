% 文件: MINROUTE.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [S,D]=minRoute(i,m,W)  % 详解: 函数定义：minRoute(i,m,W), 返回：S,D


dd=[];tt=[];ss=[];ss(1,1)=i;V=1:m;V(i)=[];dd=[0;i];  % 详解: 赋值：计算表达式并保存到 dd
kk=2;[mdd,ndd]=size(dd);  % 详解: 赋值：计算表达式并保存到 kk
while ~isempty(V)  % 详解: while 循环：当 (~isempty(V)) 为真时迭代
   [tmpd,j]=min(W(i,V));tmpj=V(j);  % 详解: 统计：最大/最小值
   for k=2:ndd  % 详解: for 循环：迭代变量 k 遍历 2:ndd
      [tmp1,jj]=min(dd(1,k)+W(dd(2,k),V));  % 详解: 统计：最大/最小值
      tmp2=V(jj);tt(k-1,:)=[tmp1,tmp2,jj];  % 详解: 赋值：将 V(...) 的结果保存到 tmp2
   end  % 详解: 执行语句
   tmp=[tmpd,tmpj,j;tt];[tmp3,tmp4]=min(tmp(:,1));  % 详解: 赋值：计算表达式并保存到 tmp
   if tmp3==tmpd, ss(1:2,kk)=[i;tmp(tmp4,2)];  % 详解: 条件判断：if (tmp3==tmpd, ss(1:2,kk)=[i;tmp(tmp4,2)];)
   else,tmp5=find(ss(:,tmp4)~=0);tmp6=length(tmp5);  % 详解: 条件判断：else 分支
      if dd(2,tmp4)==ss(tmp6,tmp4)  % 详解: 条件判断：if (dd(2,tmp4)==ss(tmp6,tmp4))
         ss(1:tmp6+1,kk)=[ss(tmp5,tmp4);tmp(tmp4,2)];  % 详解: 执行语句
         else, ss(1:3,kk)=[i;dd(2,tmp4);tmp(tmp4,2)];  % 详解: 条件判断：else 分支
   end;end  % 详解: 执行语句
   dd=[dd,[tmp3;tmp(tmp4,2)]];V(tmp(tmp4,3))=[];  % 详解: 赋值：计算表达式并保存到 dd
   [mdd,ndd]=size(dd);kk=kk+1;  % 详解: 获取向量/矩阵尺寸
end; S=ss; D=dd(1,:);  % 详解: 执行语句




