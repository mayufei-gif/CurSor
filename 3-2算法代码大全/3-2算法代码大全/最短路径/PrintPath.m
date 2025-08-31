% 文件: PrintPath.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function y=PrintPath(Min_Distance,Path,StartPointNo,EndPointNo)  % 详解: 执行语句
              i=EndPointNo;  % 详解: 赋值：计算表达式并保存到 i
              if (Path(i)==StartPointNo) & (Min_Distance(i)<Inf)  % 详解: 条件判断：if ((Path(i)==StartPointNo) & (Min_Distance(i)<Inf))
                  fprintf('起始点(%d)到终止点(%d)的路径为:',StartPointNo,i)  % 详解: 调用函数：fprintf('起始点(%d)到终止点(%d)的路径为:',StartPointNo,i)
                  fprintf('%d<-',i)  % 详解: 调用函数：fprintf('%d<-',i)
                  fprintf('%d',StartPointNo)  % 详解: 调用函数：fprintf('%d',StartPointNo)
                  fprintf('\n')  % 详解: 调用函数：fprintf('\n')
              elseif  (Min_Distance(i)==Inf)  % 详解: 条件判断：elseif ((Min_Distance(i)==Inf))
                  fprintf('起始点(%d)到终止点(%d)的路径为:空\n',StartPointNo,i)  % 详解: 调用函数：fprintf('起始点(%d)到终止点(%d)的路径为:空\n',StartPointNo,i)
              else  % 详解: 条件判断：else 分支
                   fprintf('起始点(%d)到终止点(%d)点的路径为:',StartPointNo,i)  % 详解: 调用函数：fprintf('起始点(%d)到终止点(%d)点的路径为:',StartPointNo,i)
                   fprintf('%d',i)  % 详解: 调用函数：fprintf('%d',i)
                  while(Path(i)~=StartPointNo)  % 详解: 调用函数：while(Path(i)~=StartPointNo)
                       fprintf('<-%d', Path(i))  % 详解: 调用函数：fprintf('<-%d', Path(i))
                       i=Path(i);  % 详解: 赋值：将 Path(...) 的结果保存到 i
                   end  % 详解: 执行语句
                  fprintf('<-%d',StartPointNo)  % 详解: 调用函数：fprintf('<-%d',StartPointNo)
                   fprintf('\n')  % 详解: 调用函数：fprintf('\n')
               end  % 详解: 执行语句



