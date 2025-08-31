% 文件: main.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%%使用本函数，首先要输入邻接矩阵Cost(i,j),
%%然后调用ShortestPath_Djk()求出最短路，并用PrintPath()
%%将路径打印出来。


%调用ShortestPath_Djk()和PrintPath() 来计算出最路和显示出路径
%a为（CrossPointNo）维向量，如果CrossPointNo=6
%则为6维向量。其中，a(i)表示第i点和起始点S之间的距离。
%b为算法迭代过程中，一个节点与起始点的距离被那个节点更新的，
%如果b(i)=j,表示第i个点最终由j节点修改.
[a b]=ShortestPath_Djk(Cost,CrossPointNo,s);  % 详解: 执行语句
   PrintPath(a,b,s,e);  % 详解: 调用函数：PrintPath(a,b,s,e)
  fprintf('路径长度:%f',a(e));  % 详解: 调用函数：fprintf('路径长度:%f',a(e))
  fprintf('\n');  % 详解: 调用函数：fprintf('\n')




