% 文件: Topsis.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%高校研究生院的TOPSIS评价
%课件例题

clear  % 详解: 执行语句
a=[0.1 5 5000 4.7  % 详解: 赋值：计算表达式并保存到 a
   0.2 6 6000 5.6];  % 详解: 执行语句
[m,n]=size(a);  % 详解: 获取向量/矩阵尺寸
qujian=[5 ,6];lb=2;ub=12;  % 详解: 赋值：计算表达式并保存到 qujian
a(:,2)=intervaltransfer(qujian,lb,ub,a(:,2));  % 详解: 调用函数：a(:,2)=intervaltransfer(qujian,lb,ub,a(:,2))
for j=1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
    b(:,j)=a(:,j)/norm(a(:,j));  % 详解: 调用函数：b(:,j)=a(:,j)/norm(a(:,j))
end  % 详解: 执行语句
w=[0.2,0.3,0.4,0.1];  % 详解: 赋值：计算表达式并保存到 w
c=b.*repmat(w,m,1);  % 详解: 赋值：计算表达式并保存到 c
cstar=max(c);  % 详解: 赋值：将 max(...) 的结果保存到 cstar
cstar(4)=min(c(:,4));  % 详解: 调用函数：cstar(4)=min(c(:,4))
c0=min(c);  % 详解: 赋值：将 min(...) 的结果保存到 c0
c0(4)=max(c(:,4));  % 详解: 调用函数：c0(4)=max(c(:,4))
for i=1:m  % 详解: for 循环：迭代变量 i 遍历 1:m
    sstar(i)=norm(c(i,:)-cstar);  % 详解: 调用函数：sstar(i)=norm(c(i,:)-cstar)
    s0(i)=norm(c(i,:)-c0);  % 详解: 调用函数：s0(i)=norm(c(i,:)-c0)
end  % 详解: 执行语句
f=s0./(sstar+s0);  % 详解: 赋值：计算表达式并保存到 f
[sf,ind]=sort(f,'descend');  % 详解: 执行语句
sf  % 详解: 执行语句





