% 文件: getpath.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function p=getpath(Min_Distance,Path,StartPointNo,EndPointNo)  % 详解: 执行语句
              i=EndPointNo;np=0;p=[];  % 详解: 赋值：计算表达式并保存到 i
              if (Path(i)==StartPointNo) & (Min_Distance(i)<Inf)  % 详解: 条件判断：if ((Path(i)==StartPointNo) & (Min_Distance(i)<Inf))
                  np=1;p(1)=i;  % 详解: 赋值：计算表达式并保存到 np
                  np=2;p(2)=StartPointNo;  % 详解: 赋值：计算表达式并保存到 np
                  fprintf('\n')  % 详解: 调用函数：fprintf('\n')
              elseif  (Min_Distance(i)==Inf)  % 详解: 条件判断：elseif ((Min_Distance(i)==Inf))
                  fprintf('起始点(%d)到终止点(%d)的路径为:空\n',StartPointNo,i)  % 详解: 调用函数：fprintf('起始点(%d)到终止点(%d)的路径为:空\n',StartPointNo,i)
              else  % 详解: 条件判断：else 分支
                  np=1;p(1)=i;  % 详解: 赋值：计算表达式并保存到 np
                  while(Path(i)~=StartPointNo)  % 详解: 调用函数：while(Path(i)~=StartPointNo)
                       np=np+1;p(np)=Path(i);  % 详解: 赋值：计算表达式并保存到 np
                       i=Path(i);  % 详解: 赋值：将 Path(...) 的结果保存到 i
                   end  % 详解: 执行语句
                  np=np+1;p(np)=StartPointNo;  % 详解: 赋值：计算表达式并保存到 np
                   fprintf('\n')  % 详解: 调用函数：fprintf('\n')
               end  % 详解: 执行语句
               
               
               n=length(p);  % 详解: 赋值：将 length(...) 的结果保存到 n
               if n~=0  % 详解: 条件判断：if (n~=0)
                   q=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 q
                   for k=1:n  % 详解: for 循环：迭代变量 k 遍历 1:n
                        q(k)=p(n+1-k);  % 详解: 调用函数：q(k)=p(n+1-k)
                   end  % 详解: 执行语句
                   p=q;  % 详解: 赋值：计算表达式并保存到 p
               end  % 详解: 执行语句
                   
              



