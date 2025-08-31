% 文件: KRUSKAL.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [out,len]=kruskal(map)  % 详解: 函数定义：kruskal(map), 返回：out,len


mx=0;  % 详解: 赋值：计算表达式并保存到 mx
[m,n]=size(map);  % 详解: 获取向量/矩阵尺寸
mx=max(max(map(:,1:2)));  % 详解: 赋值：将 max(...) 的结果保存到 mx
op=10000*ones(mx,mx);  % 详解: 赋值：计算表达式并保存到 op
for i=1:m  % 详解: for 循环：迭代变量 i 遍历 1:m
   op(map(i,1),map(i,2))=map(i,3);  % 详解: 调用函数：op(map(i,1),map(i,2))=map(i,3)
   op(map(i,2),map(i,1))=map(i,3);  % 详解: 调用函数：op(map(i,2),map(i,1))=map(i,3)
end  % 详解: 执行语句
[out,len]=krusk(op);  % 详解: 执行语句
out=out';  % 赋值：设置变量 out  % 详解: 赋值：计算表达式并保存到 out  % 详解: 赋值：计算表达式并保存到 out

close;  % 详解: 执行语句
 set(gcf,'numbertitle','off');  % 详解: 调用函数：set(gcf,'numbertitle','off')
 set(gcf,'name','Kruskal');  % 详解: 调用函数：set(gcf,'name','Kruskal')
 set(gca,'visible','off');  % 详解: 调用函数：set(gca,'visible','off')
 axis square;  % 详解: 执行语句
 hold on;  % 详解: 执行语句
 
 b=linspace(0,2*pi,mx+1);  % 详解: 赋值：将 linspace(...) 的结果保存到 b
 b1=10*sin(b);  % 详解: 赋值：计算表达式并保存到 b1
 b2=10*cos(b);  % 详解: 赋值：计算表达式并保存到 b2
 plot(b1,b2,'ko');  % 详解: 调用函数：plot(b1,b2,'ko')
 
 hh=char(49:48+mx);  % 详解: 赋值：将 char(...) 的结果保存到 hh
 for i=1:mx  % 详解: for 循环：迭代变量 i 遍历 1:mx
    text(b1(i)+1,b2(i),hh(i));  % 详解: 调用函数：text(b1(i)+1,b2(i),hh(i))
 end;  % 详解: 执行语句
 map  % 详解: 执行语句
 for j=1:m  % 详解: for 循环：迭代变量 j 遍历 1:m
   for i=1:2  % 详解: for 循环：迭代变量 i 遍历 1:2
     c1(i)=b1(map(j,i));  % 详解: 调用函数：c1(i)=b1(map(j,i))
     c2(i)=b2(map(j,i));  % 详解: 调用函数：c2(i)=b2(map(j,i))
    end;  % 详解: 执行语句
    plot(c1,c2,':');  % 详解: 调用函数：plot(c1,c2,':')
end;  % 详解: 执行语句

 k1=length(out);  % 详解: 赋值：将 length(...) 的结果保存到 k1
 for i=1:k1  % 详解: for 循环：迭代变量 i 遍历 1:k1
   for j=1:2  % 详解: for 循环：迭代变量 j 遍历 1:2
      d1(j)=b1(out(i,j));  % 详解: 调用函数：d1(j)=b1(out(i,j))
      d2(j)=b2(out(i,j));  % 详解: 调用函数：d2(j)=b2(out(i,j))
   end  % 详解: 执行语句
   h=plot(d1,d2,'r');  % 详解: 赋值：将 plot(...) 的结果保存到 h
   set(h,'linewidth',2);  % 详解: 调用函数：set(h,'linewidth',2)
 end  % 详解: 执行语句
 legend(h,'粗线为最优树');  % 详解: 调用函数：legend(h,'粗线为最优树')
 hold off  % 详解: 执行语句



