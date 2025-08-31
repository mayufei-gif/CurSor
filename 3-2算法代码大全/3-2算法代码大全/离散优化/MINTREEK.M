% 文件: MINTREEK.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [Wt,Pp]=mintreek(n,W)  % 详解: 函数定义：mintreek(n,W), 返回：Wt,Pp


tmpa=find(W~=inf);[tmpb,tmpc]=find(W~=inf);  % 详解: 赋值：将 find(...) 的结果保存到 tmpa
w=W(tmpa);e=[tmpb,tmpc];  % 详解: 赋值：将 W(...) 的结果保存到 w
[wa,wb]=sort(w);E=[e(wb,:),wa,wb];[nE,mE]=size(E);  % 详解: 获取向量/矩阵尺寸
temp=find(E(:,1)-E(:,2));E=E(temp,:);  % 详解: 赋值：将 find(...) 的结果保存到 temp
P=E(1,:);k=length(E(:,1));  % 详解: 赋值：将 E(...) 的结果保存到 P
while (rank(E)>0)  % 详解: while 循环：当 ((rank(E)>0)) 为真时迭代
  temp1=max(E(1,2),E(1,1));temp2=min(E(1,2),E(1,1));  % 详解: 赋值：将 max(...) 的结果保存到 temp1
  for i=1:k;  % 详解: for 循环：迭代变量 i 遍历 1:k;
    if (E(i,1)==temp1), E(i,1)=temp2; end;  % 详解: 条件判断：if ((E(i,1)==temp1), E(i,1)=temp2; end;)
if (E(i,2)==temp1), E(i,2)=temp2; end;  % 详解: 条件判断：if ((E(i,2)==temp1), E(i,2)=temp2; end;)
  end;  % 详解: 执行语句
  a=find(E(:,1)-E(:,2));E=E(a,:);  % 详解: 赋值：将 find(...) 的结果保存到 a
  if (rank(E)>0),P=[P;E(1,:)];k=length(E(:,1)); end;  % 详解: 条件判断：if ((rank(E)>0),P=[P;E(1,:)];k=length(E(:,1)); end;)
end;  % 详解: 执行语句
Wt=sum(P(:,3));Pp=[e(P(:,4),:),P(:,3:4)];  % 详解: 赋值：将 sum(...) 的结果保存到 Wt
for i=1:length(P(:,3));  % 详解: for 循环：迭代变量 i 遍历 1:length(P(:,3));
disp(['   ','e',num2str(P(i,4)),' ','(v',...  % 详解: 打印/显示输出
num2str(P(i,1)),' ','v',num2str(P(i,2)),')']);  % 详解: 调用函数：num2str(P(i,1)),' ','v',num2str(P(i,2)),')'])
end;  % 详解: 执行语句
axis equal; hold on  % 详解: 执行语句
[x,y]=cylinder(1,n);xm=min(x(1,:)); ym=min(y(1,:));  % 详解: 统计：最大/最小值
xx=max(x(1,:)); yy=max(y(1,:));  % 详解: 赋值：将 max(...) 的结果保存到 xx
axis([xm-abs(xm)*0.15,xx+abs(xx)*0.15,ym-abs(ym)*0.15, yy+abs(yy)*0.15]); plot(x(1,:),y(1,:),'ko')  % 详解: 调用函数：axis([xm-abs(xm)*0.15,xx+abs(xx)*0.15,ym-abs(ym)*0.15, yy+abs(yy)*0.15]); plot(x(1,:),y(1,:),'ko')
for i=1:n; temp=['  v',int2str(i)];  % 详解: for 循环：迭代变量 i 遍历 1:n; temp=['  v',int2str(i)];
   text(x(1,i),y(1,i),temp); end;  % 详解: 执行语句
for i=1:nE; plot(x(1,e(i,:)),y(1,e(i,:)),'b'); end;  % 详解: for 循环：迭代变量 i 遍历 1:nE; plot(x(1,e(i,:)),y(1,e(i,:)),'b'); end;
for i=1:length(P(:,4));  % 详解: for 循环：迭代变量 i 遍历 1:length(P(:,4));
  plot(x(1,Pp(i,1:2)),y(1,Pp(i,1:2)),'r'); end;  % 详解: 绘图：二维曲线
text(-0.35,-1.2,['最小生成树的权为',' ',num2str(Wt)]);  % 详解: 调用函数：text(-0.35,-1.2,['最小生成树的权为',' ',num2str(Wt)])
title('红色连线为最小生成树'); axis('off');hold off  % 详解: 图形标注/图例





