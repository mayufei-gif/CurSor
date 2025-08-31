% 文件: TSP_SA.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clc,clear  % 详解: 执行语句
load berlin52.tsp  % 详解: 执行语句
berlin52(:,1)=[];  % 详解: 执行语句
x=berlin52(:,1);  % 详解: 赋值：将 berlin52(...) 的结果保存到 x
y=berlin52(:,2);  % 详解: 赋值：将 berlin52(...) 的结果保存到 y
d=zeros(52);  % 详解: 赋值：将 zeros(...) 的结果保存到 d
for i=1:52  % 详解: for 循环：迭代变量 i 遍历 1:52
for j=1:52  % 详解: for 循环：迭代变量 j 遍历 1:52
d(i,j)=sqrt((x(i)-x(j)).^2+(y(i)-y(j)).^2);  % 详解: 调用函数：d(i,j)=sqrt((x(i)-x(j)).^2+(y(i)-y(j)).^2)
end  % 详解: 执行语句
end  % 详解: 执行语句

figure(1)  % 详解: 调用函数：figure(1)
plot(x,y,'o')  % 详解: 调用函数：plot(x,y,'o')
hold on  % 详解: 执行语句

S0=[];Sum=inf;  % 详解: 赋值：计算表达式并保存到 S0
rand('state',sum(clock));  % 详解: 调用函数：rand('state',sum(clock))
for j=1:1000  % 详解: for 循环：迭代变量 j 遍历 1:1000
S=[1 1+randperm(51),52];  % 详解: 赋值：计算表达式并保存到 S
temp=0;  % 详解: 赋值：计算表达式并保存到 temp
for i=1:52  % 详解: for 循环：迭代变量 i 遍历 1:52
temp=temp+d(S(i),S(i+1));  % 详解: 赋值：计算表达式并保存到 temp
end  % 详解: 执行语句
if temp<Sum  % 详解: 条件判断：if (temp<Sum)
S0=S;Sum=temp;  % 详解: 赋值：计算表达式并保存到 S0
end  % 详解: 执行语句
end  % 详解: 执行语句
e=0.1^30;L=2000;at=0.999;T=1;  % 详解: 赋值：计算表达式并保存到 e
for k=1:L  % 详解: for 循环：迭代变量 k 遍历 1:L
    k  % 详解: 执行语句
c=2+floor(50*rand(2,1));  % 详解: 赋值：计算表达式并保存到 c
c=sort(c);  % 详解: 赋值：将 sort(...) 的结果保存到 c
c1=c(1);c2=c(2);  % 详解: 赋值：将 c(...) 的结果保存到 c1
df=d(S0(c1-1),S0(c2))+d(S0(c1),S0(c2+1))-d(S0(c1-1),S0(c1))-d(S0(c2),S0(c2+1));  % 详解: 赋值：将 d(...) 的结果保存到 df
if df<0  % 详解: 条件判断：if (df<0)
S0=[S0(1:c1-1),S0(c2:-1:c1),S0(c2+1:53)];  % 详解: 赋值：计算表达式并保存到 S0
Sum=Sum+df;  % 详解: 赋值：计算表达式并保存到 Sum
elseif exp(-df/T)>rand(1)  % 详解: 条件判断：elseif (exp(-df/T)>rand(1))
S0=[S0(1:c1-1),S0(c2:-1:c1),S0(c2+1:53)];  % 详解: 赋值：计算表达式并保存到 S0
Sum=Sum+df;  % 详解: 赋值：计算表达式并保存到 Sum
end  % 详解: 执行语句
T=T*at;  % 详解: 赋值：计算表达式并保存到 T
if T<e  % 详解: 条件判断：if (T<e)
break;  % 详解: 跳出循环：break
end  % 详解: 执行语句
plot(x(S0),y(S0),'--')  % 详解: 调用函数：plot(x(S0),y(S0),'--')
hold on  % 详解: 执行语句
end  % 详解: 执行语句
S0,Sum  % 详解: 执行语句
figure(2)  % 详解: 调用函数：figure(2)
plot(x,y,'*',x(S0),y(S0),'-')  % 详解: 调用函数：plot(x,y,'*',x(S0),y(S0),'-')





