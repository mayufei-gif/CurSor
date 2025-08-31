% 文件: ShortestPath_Djk.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%function [Min_Dist,Muti_Path]=ShortestPath_Djk(Cost,CrossPointNo,StartPoint)
%%%Creat Graph
%%%Cost is lingjie matrix,defaut value is inf
%%%The total Number is CrossPointNo
%%%StartPoint is the inicial Point
function [Min_Distance,Path]=ShortestPath_Djk(Cost,CrossPointNo,StartPoint)  % 详解: 函数定义：ShortestPath_Djk(Cost,CrossPointNo,StartPoint), 返回：Min_Distance,Path
for i=1:CrossPointNo  % 详解: for 循环：迭代变量 i 遍历 1:CrossPointNo
    for j=1:CrossPointNo  % 详解: for 循环：迭代变量 j 遍历 1:CrossPointNo
        Min_Dist(i,j)=Cost(i,j);  % 详解: 调用函数：Min_Dist(i,j)=Cost(i,j)
        Muti_Path(i,j)=StartPoint;  % 详解: 执行语句
        IsFinal(i,j)=0;  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句
IsFinal(StartPoint,StartPoint)=1;  % 详解: 执行语句

   for j=1:(CrossPointNo-1)  % 详解: for 循环：迭代变量 j 遍历 1:(CrossPointNo-1)
        
       MinPathDist=inf;  % 详解: 赋值：计算表达式并保存到 MinPathDist
        for temp_w=1:CrossPointNo  % 详解: for 循环：迭代变量 temp_w 遍历 1:CrossPointNo
		
		       if (IsFinal(StartPoint,temp_w)==0) & (Min_Dist(StartPoint,temp_w)< MinPathDist)  % 详解: 条件判断：if ((IsFinal(StartPoint,temp_w)==0) & (Min_Dist(StartPoint,temp_w)< MinPathDist))
                           temp_v=temp_w;  % 详解: 赋值：计算表达式并保存到 temp_v
                           MinPathDist=Min_Dist(StartPoint,temp_v);  % 详解: 赋值：将 Min_Dist(...) 的结果保存到 MinPathDist
               end  % 详解: 执行语句
              
         end  % 详解: 执行语句
         IsFinal(StartPoint,temp_v)=1;  % 详解: 执行语句
             for temp_z=1:CrossPointNo  % 详解: for 循环：迭代变量 temp_z 遍历 1:CrossPointNo
                  if (IsFinal(StartPoint,temp_z)==0) &( (MinPathDist+Cost(temp_v,temp_z))<(Cost(StartPoint,temp_z)))  % 详解: 条件判断：if ((IsFinal(StartPoint,temp_z)==0) &( (MinPathDist+Cost(temp_v,temp_z))<(Cost(StartPoint,temp_z))))
                           Cost(StartPoint,temp_z)=(MinPathDist+Cost(temp_v,temp_z));  % 详解: 调用函数：Cost(StartPoint,temp_z)=(MinPathDist+Cost(temp_v,temp_z))
                           Min_Dist(StartPoint,temp_z)=Cost(StartPoint,temp_z);  % 详解: 调用函数：Min_Dist(StartPoint,temp_z)=Cost(StartPoint,temp_z)
                           Muti_Path(StartPoint,temp_z)=temp_v;  % 详解: 执行语句
                  end  % 详解: 执行语句
             end  % 详解: 执行语句
     end  % 详解: 执行语句
     Min_Distance= Min_Dist(StartPoint,:);  % 详解: 赋值：将 Min_Dist(...) 的结果保存到 Min_Distance
     Path=Muti_Path(StartPoint,:);  % 详解: 赋值：将 Muti_Path(...) 的结果保存到 Path
     
     
     
     
         
                   
    
    
   















