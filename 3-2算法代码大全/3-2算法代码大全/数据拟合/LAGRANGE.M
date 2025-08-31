% 文件: LAGRANGE.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function yy=lagrange(x,y,xx)  % 详解: 执行语句


m=length(x);n=length(y);  % 详解: 赋值：将 length(...) 的结果保存到 m
if m~=n, error('向量x与y的长度必须一致');end  % 详解: 条件判断：if (m~=n, error('向量x与y的长度必须一致');end)
s=0;  % 详解: 赋值：计算表达式并保存到 s
for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
   t=ones(1,length(xx));  % 详解: 赋值：将 ones(...) 的结果保存到 t
   for j=1:n  % 详解: for 循环：迭代变量 j 遍历 1:n
      if j~=i,  % 详解: 条件判断：if (j~=i,)
         t=t.*(xx-x(j))/(x(i)-x(j));  % 详解: 赋值：计算表达式并保存到 t
      end  % 详解: 执行语句
   end  % 详解: 执行语句
   s=s+t*y(i);  % 详解: 赋值：计算表达式并保存到 s
end  % 详解: 执行语句
yy=s;  % 详解: 赋值：计算表达式并保存到 yy

   



