% 文件: DIJKSTRA.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [p,v]=dijkstra(map,u1,u2)  % 详解: 函数定义：dijkstra(map,u1,u2), 返回：p,v


[m,n]=size(map);  % 详解: 获取向量/矩阵尺寸
mx=max(max(map(:,1:2)));  % 详解: 赋值：将 max(...) 的结果保存到 mx
l=10000*ones(mx,mx);  % 详解: 赋值：计算表达式并保存到 l
for i=1:m  % 详解: for 循环：迭代变量 i 遍历 1:m
    l(map(i,1),map(i,2))=map(i,3);  % 详解: 调用函数：l(map(i,1),map(i,2))=map(i,3)
    l(map(i,2),map(i,1))=map(i,3);  % 详解: 调用函数：l(map(i,2),map(i,1))=map(i,3)
 end;  % 详解: 执行语句
 [p,v]=dijk(l,u1,u2);  % 详解: 执行语句
p=p(p~=0);  % 详解: 赋值：将 p(...) 的结果保存到 p

close;  % 详解: 执行语句
 set(gcf,'numbertitle','off');  % 详解: 调用函数：set(gcf,'numbertitle','off')
 set(gcf,'name','Dijkstra');  % 详解: 调用函数：set(gcf,'name','Dijkstra')
 set(gca,'visible','off');  % 详解: 调用函数：set(gca,'visible','off')
 axis square;  % 详解: 执行语句
 hold on;  % 详解: 执行语句
 
 b=linspace(0,2*pi,mx+1);  % 详解: 赋值：将 linspace(...) 的结果保存到 b
 b1=10*sin(b);  % 详解: 赋值：计算表达式并保存到 b1
 b2=10*cos(b);  % 详解: 赋值：计算表达式并保存到 b2
 plot(b1,b2,'ko');  % 详解: 调用函数：plot(b1,b2,'ko')
 hh=char(49:48+mx);  % 详解: 赋值：将 char(...) 的结果保存到 hh
 for i=1:mx  % 详解: for 循环：迭代变量 i 遍历 1:mx
    text(b1(i)+0.5,b2(i),hh(i));  % 详解: 调用函数：text(b1(i)+0.5,b2(i),hh(i))
 end;  % 详解: 执行语句
 for j=1:m  % 详解: for 循环：迭代变量 j 遍历 1:m
   for i=1:2  % 详解: for 循环：迭代变量 i 遍历 1:2
    c1(i)=b1(map(j,i));  % 详解: 调用函数：c1(i)=b1(map(j,i))
    c2(i)=b2(map(j,i));  % 详解: 调用函数：c2(i)=b2(map(j,i))
    end;  % 详解: 执行语句
    plot(c1,c2,':');  % 详解: 调用函数：plot(c1,c2,':')
  end;  % 详解: 执行语句

 kk=length(p);  % 详解: 赋值：将 length(...) 的结果保存到 kk
 k=0;  % 详解: 赋值：计算表达式并保存到 k
 for i=1:kk  % 详解: for 循环：迭代变量 i 遍历 1:kk
    if(p(1,i)~=0)  % 详解: 调用函数：if(p(1,i)~=0)
       k=k+1;  % 详解: 赋值：计算表达式并保存到 k
    end;  % 详解: 执行语句
 end;  % 详解: 执行语句
 for i=1:k  % 详解: for 循环：迭代变量 i 遍历 1:k
   d1(i)=b1(p(1,i));  % 详解: 调用函数：d1(i)=b1(p(1,i))
   d2(i)=b2(p(1,i));  % 详解: 调用函数：d2(i)=b2(p(1,i))
  h=plot(d1,d2,'r');  % 详解: 赋值：将 plot(...) 的结果保存到 h
end;  % 详解: 执行语句
 set(h,'linewidth',2);  % 详解: 调用函数：set(h,'linewidth',2)
 legend(h,'粗线表示最短路');  % 详解: 调用函数：legend(h,'粗线表示最短路')
 hold off  % 详解: 执行语句


 




