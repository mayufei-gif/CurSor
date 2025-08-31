% 文件: kmeans1.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [Yc,c,errlog]=kmeans1(Y,K,maxiter)  % 详解: 函数定义：kmeans1(Y,K,maxiter), 返回：Yc,c,errlog
  [M, N]=size(Y);  % 详解: 获取向量/矩阵尺寸
  if  (K>M)  % 详解: 条件判断：if ((K>M))
      error('MORE CENTROID than data vectors.')  % 详解: 调用函数：error('MORE CENTROID than data vectors.')
  end  % 详解: 执行语句
  errlog=zeros(maxiter,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 errlog
  perm=randperm(M);  % 详解: 赋值：将 randperm(...) 的结果保存到 perm
  Yc=Y(perm(1:K),:);  % 详解: 赋值：将 Y(...) 的结果保存到 Yc
  d2y=(ones(K,1)*sum((Y.^2)'))';  % 详解: 赋值：计算表达式并保存到 d2y
  for i=1:maxiter  % 详解: for 循环：迭代变量 i 遍历 1:maxiter
      Yc_old=Yc;  % 详解: 赋值：计算表达式并保存到 Yc_old
      d2=d2y+ones(M,1)*sum((Yc.^2)')-2*Y*Yc';  % 详解: 赋值：计算表达式并保存到 d2
      [errvals,c]=min(d2');  % 统计：最大/最小值  % 详解: 统计：最大/最小值  % 详解: 统计：最大/最小值
      for k=1:K  % 详解: for 循环：迭代变量 k 遍历 1:K
          if(sum(c==k)>0)  % 详解: 调用函数：if(sum(c==k)>0)
          Yc(k,:) =sum(Y(c==k,:))/sum(c==k);  % 详解: 调用函数：Yc(k,:) =sum(Y(c==k,:))/sum(c==k)
          end  % 详解: 执行语句
      end  % 详解: 执行语句
      errlog(i)=sum(errvals);  % 详解: 调用函数：errlog(i)=sum(errvals)
      fprintf(1,'...iteration%4d...Error%11.6f\n',i,errlog(i));  % 详解: 调用函数：fprintf(1,'...iteration%4d...Error%11.6f\n',i,errlog(i))
      if (max(max(abs(Yc-Yc_old)))<10*eps)  % 详解: 条件判断：if ((max(max(abs(Yc-Yc_old)))<10*eps))
          errlog=errlog(1:i);  % 详解: 赋值：将 errlog(...) 的结果保存到 errlog
          return  % 详解: 返回：从当前函数返回
      end  % 详解: 执行语句
  end  % 详解: 执行语句
      



