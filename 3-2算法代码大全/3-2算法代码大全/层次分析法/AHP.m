% 文件: AHP.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clc,clear  % 详解: 执行语句
fid=fopen('txt3.txt','r');  % 详解: 赋值：将 fopen(...) 的结果保存到 fid
n1=6;  % 详解: 赋值：计算表达式并保存到 n1
n2=3;  % 详解: 赋值：计算表达式并保存到 n2
a=[];  % 详解: 赋值：计算表达式并保存到 a
for i=1:(n1)+1  % 详解: for 循环：迭代变量 i 遍历 1:(n1)+1
    tmp=str2num(fgetl(fid));  % 详解: 赋值：将 str2num(...) 的结果保存到 tmp
    a=[a;tmp];  % 详解: 赋值：计算表达式并保存到 a
end  % 详解: 执行语句
for i=1:n1  % 详解: for 循环：迭代变量 i 遍历 1:n1
     str1=char(['b',int2str(i),'=[];']);  % 详解: 赋值：将 char(...) 的结果保存到 str1
     str2=char(['b',int2str(i),'=[b',int2str(i),';tmp];']);  % 详解: 赋值：将 char(...) 的结果保存到 str2
     eval(str1);  % 详解: 调用函数：eval(str1)
     for j=1:n2  % 详解: for 循环：迭代变量 j 遍历 1:n2
          tmp=str2num(fgetl(fid));  % 详解: 赋值：将 str2num(...) 的结果保存到 tmp
          eval(str2);  % 详解: 调用函数：eval(str2)
     end  % 详解: 执行语句
end  % 详解: 执行语句
ri=[0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45];  % 详解: 赋值：计算表达式并保存到 ri
 [x,y]=eig(a);  % 详解: 执行语句
lamda=max(diag(y));  % 详解: 赋值：将 max(...) 的结果保存到 lamda
num=find(diag(y)==lamda);  % 详解: 赋值：将 find(...) 的结果保存到 num
w0=x(:,num)/sum(x(:,num));  % 详解: 赋值：将 x(...) 的结果保存到 w0
cr0=(lamda-n1)/(n1-1)/ri(n1)  % 详解: 赋值：计算表达式并保存到 cr0
for i=1:n1  % 详解: for 循环：迭代变量 i 遍历 1:n1
    [x,y]=eig(eval(char(['b',int2str(i)])));  % 详解: 执行语句
    lamda=max(diag(y));  % 详解: 赋值：将 max(...) 的结果保存到 lamda
    num=find(diag(y)==lamda);  % 详解: 赋值：将 find(...) 的结果保存到 num
    w1(:,i)=x(:,num)/sum(x(:,num));  % 详解: 调用函数：w1(:,i)=x(:,num)/sum(x(:,num))
    cr1(i)=(lamda-n2)/(n2-1)/ri(n2);  % 详解: 调用函数：cr1(i)=(lamda-n2)/(n2-1)/ri(n2)
end  % 详解: 执行语句
cr1, ts=w1*w0, cr=cr1*w0  % 详解: 执行语句



