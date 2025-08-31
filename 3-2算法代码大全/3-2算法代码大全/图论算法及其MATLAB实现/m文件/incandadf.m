% 文件: incandadf.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function W=incandadf(G,f)  % 详解: 执行语句

if f==0  % 详解: 条件判断：if (f==0)
    m=sum(sum(G))/2;  % 详解: 赋值：将 sum(...) 的结果保存到 m
    n=size(G,1);  % 详解: 赋值：将 size(...) 的结果保存到 n
    W=zeros(n,m);  % 详解: 赋值：将 zeros(...) 的结果保存到 W
    k=1;  % 详解: 赋值：计算表达式并保存到 k
    for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
        for j=i:n  % 详解: for 循环：迭代变量 j 遍历 i:n
            if G(i,j)~=0  % 详解: 条件判断：if (G(i,j)~=0)
                W(i,k)=1;  % 详解: 执行语句
                W(j,k)=1;  % 详解: 执行语句
                k=k+1;  % 详解: 赋值：计算表达式并保存到 k
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
elseif f==1  % 详解: 条件判断：elseif (f==1)
    m=size(G,2);  % 详解: 赋值：将 size(...) 的结果保存到 m
    n=size(G,1);  % 详解: 赋值：将 size(...) 的结果保存到 n
    W=zeros(n,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 W
    for i=1:m  % 详解: for 循环：迭代变量 i 遍历 1:m
        a=find(G(:,i)~=0);  % 详解: 赋值：将 find(...) 的结果保存到 a
        W(a(1),a(2))=1;  % 详解: 执行语句
        W(a(2),a(1))=1;  % 详解: 执行语句
    end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
    fprint('please input the right value of f');  % 详解: 调用函数：fprint('please input the right value of f')
end  % 详解: 执行语句
W;  % 详解: 执行语句



