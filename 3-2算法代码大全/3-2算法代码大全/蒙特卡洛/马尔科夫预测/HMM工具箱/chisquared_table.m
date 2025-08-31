% 文件: chisquared_table.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function X2 = chisquared_table(P,v)  % 详解: 执行语句




[mP,nP]=size(P);  % 详解: 获取向量/矩阵尺寸
[mv,nv]=size(v);  % 详解: 获取向量/矩阵尺寸
if mP~=mv | nP~=nv,  % 详解: 条件判断：if (mP~=mv | nP~=nv,)
  if mP==1 & nP==1,  % 详解: 条件判断：if (mP==1 & nP==1,)
    P=P*ones(mv,nv);  % 详解: 赋值：计算表达式并保存到 P
  elseif mv==1 & nv==1,  % 详解: 条件判断：elseif (mv==1 & nv==1,)
    v=v*ones(mP,nP);  % 详解: 赋值：计算表达式并保存到 v
  else  % 详解: 条件判断：else 分支
    error('P and v must be the same size')  % 详解: 调用函数：error('P and v must be the same size')
  end  % 详解: 执行语句
end  % 详解: 执行语句
[m,n]=size(P);  X2 = zeros(m,n);  % 详解: 创建零矩阵/数组
for i=1:m,  % 详解: for 循环：迭代变量 i 遍历 1:m,
 for j=1:n,  % 详解: for 循环：迭代变量 j 遍历 1:n,
  if v(i,j)<=10,  % 详解: 条件判断：if (v(i,j)<=10,)
   x0=P(i,j)*v(i,j);  % 详解: 赋值：将 P(...) 的结果保存到 x0
  else  % 详解: 条件判断：else 分支
   x0=v(i,j);  % 详解: 赋值：将 v(...) 的结果保存到 x0
  end  % 详解: 执行语句
   X2(i,j) = fsolve('chiaux',x0,zeros(16,1),[],[v(i,j),P(i,j)]);  % 详解: 调用函数：X2(i,j) = fsolve('chiaux',x0,zeros(16,1),[],[v(i,j),P(i,j)])
 end  % 详解: 执行语句
end  % 详解: 执行语句




