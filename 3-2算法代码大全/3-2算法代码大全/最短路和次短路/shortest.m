% 文件: shortest.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%function p=shortest(startp,endp,Cost)
function p=shortest(startp,endp,Cost)  % 详解: 执行语句
CrossPointNo=length(Cost);  % 详解: 赋值：将 length(...) 的结果保存到 CrossPointNo
[a b]=Shortest_Djk(Cost,CrossPointNo,startp);  % 详解: 执行语句
  p=getpath(a,b,startp,endp)  % 详解: 赋值：将 getpath(...) 的结果保存到 p
  fprintf('路径长度:%f',a(endp));  % 详解: 调用函数：fprintf('路径长度:%f',a(endp))
  fprintf('\n');  % 详解: 调用函数：fprintf('\n')




