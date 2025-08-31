% 文件: secshortest.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%function sp=secshortest(startp,endp,p,c)
%求解第二条最短路，p为最短路径，c为邻接矩阵
%canshu.m文件提供了一个邻接矩阵的实例
%p需先由shortest函数求出
function sp=secshortest(startp,endp,p,c)  % 详解: 执行语句
n=length(p);  % 详解: 赋值：将 length(...) 的结果保存到 n
npp=0;pp=[];  % 详解: 赋值：计算表达式并保存到 npp
j=-1;  % 详解: 赋值：计算表达式并保存到 j
while j~=n-3  % 详解: while 循环：当 (j~=n-3) 为真时迭代
    j=j+1;  % 详解: 赋值：计算表达式并保存到 j
    c(p(n-j-1),p(n-j))=inf;  % 详解: 执行语句
    p0=shortest(startp,p(n-j),c);  % 详解: 赋值：将 shortest(...) 的结果保存到 p0
    n0=length(p0);  % 详解: 赋值：将 length(...) 的结果保存到 n0
    if j==0  % 详解: 条件判断：if (j==0)
        r=[];nr=0;  % 详解: 赋值：计算表达式并保存到 r
    else  % 详解: 条件判断：else 分支
        nr=nr+1;  % 详解: 赋值：计算表达式并保存到 nr
        r(nr)=p(n+1-j);  % 详解: 调用函数：r(nr)=p(n+1-j)
    end  % 详解: 执行语句
    if r==[]  % 详解: 条件判断：if (r==[])
        npp=1;pp(npp,:)=p0;  % 详解: 赋值：计算表达式并保存到 npp
    else  % 详解: 条件判断：else 分支
        store=p0;  % 详解: 赋值：计算表达式并保存到 store
        for i=1:nr  % 详解: for 循环：迭代变量 i 遍历 1:nr
             store(n0+i)=r(nr+1-i);  % 详解: 调用函数：store(n0+i)=r(nr+1-i)
         end  % 详解: 执行语句
         npp=npp+1;  % 详解: 赋值：计算表达式并保存到 npp
         pp(npp,1:length(store))=store;  % 详解: 获取向量/矩阵尺寸
     end  % 详解: 执行语句
 end  % 详解: 执行语句
 
 np3=0;p3=[];  % 详解: 赋值：计算表达式并保存到 np3
 for i=1:npp  % 详解: for 循环：迭代变量 i 遍历 1:npp
     
     if pp(i,length(pp(i,:)))~=0  % 详解: 条件判断：if (pp(i,length(pp(i,:)))~=0)
         l=length(pp(i,:));  % 详解: 赋值：将 length(...) 的结果保存到 l
     else  % 详解: 条件判断：else 分支
         for l=1:length(pp(i,:))  % 详解: for 循环：迭代变量 l 遍历 1:length(pp(i,:))
             if pp(i,l)==0  % 详解: 条件判断：if (pp(i,l)==0)
                 break;  % 详解: 跳出循环：break
             end  % 详解: 执行语句
         end  % 详解: 执行语句
         l=l-1;  % 详解: 赋值：计算表达式并保存到 l
     end  % 详解: 执行语句
     
     store=pp(i,1:l);  % 详解: 赋值：将 pp(...) 的结果保存到 store
     if roadcost(store,c)>=roadcost(p,c)  % 详解: 条件判断：if (roadcost(store,c)>=roadcost(p,c))
         np3=np3+1;  % 详解: 赋值：计算表达式并保存到 np3
         p3(np3,1:length(store))=store;  % 详解: 获取向量/矩阵尺寸
     end  % 详解: 执行语句
 end  % 详解: 执行语句
 
 
 sp=[];nsp=0;  % 详解: 赋值：计算表达式并保存到 sp
 for i=1:np3  % 详解: for 循环：迭代变量 i 遍历 1:np3
     if p3(i,length(p3(i,:)))~=0  % 详解: 条件判断：if (p3(i,length(p3(i,:)))~=0)
         l=length(p3(i,:));  % 详解: 赋值：将 length(...) 的结果保存到 l
     else  % 详解: 条件判断：else 分支
         for l=1:length(p3(i,:))  % 详解: for 循环：迭代变量 l 遍历 1:length(p3(i,:))
             if p3(i,l)==0  % 详解: 条件判断：if (p3(i,l)==0)
                 break;  % 详解: 跳出循环：break
             end  % 详解: 执行语句
         end  % 详解: 执行语句
         l=l-1;  % 详解: 赋值：计算表达式并保存到 l
     end  % 详解: 执行语句
     store=p3(i,1:l);  % 详解: 赋值：将 p3(...) 的结果保存到 store
     
     if i==1  % 详解: 条件判断：if (i==1)
         nsp=1;  % 详解: 赋值：计算表达式并保存到 nsp
         sp(1,1:l)=store;  % 详解: 执行语句
     else  % 详解: 条件判断：else 分支
         if roadcost(store,c)<roadcost(sp(1,:),c)  % 详解: 条件判断：if (roadcost(store,c)<roadcost(sp(1,:),c))
             sp=[];  % 详解: 赋值：计算表达式并保存到 sp
             nsp=1;  % 详解: 赋值：计算表达式并保存到 nsp
             sp(1,1:l)=store;  % 详解: 执行语句
         end  % 详解: 执行语句
         if roadcost(store,c)==roadcost(sp(1,:),c)  % 详解: 条件判断：if (roadcost(store,c)==roadcost(sp(1,:),c))
             nsp=nsp+1;  % 详解: 赋值：计算表达式并保存到 nsp
             sp(nsp,1:l)=store;  % 详解: 执行语句
         end  % 详解: 执行语句
     end  % 详解: 执行语句
 end  % 详解: 执行语句
 
         
     



