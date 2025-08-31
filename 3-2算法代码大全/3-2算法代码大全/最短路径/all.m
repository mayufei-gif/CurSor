% 文件: all.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

Muti_Cost=zeros(CrossPointNo,CrossPointNo);  % 详解: 赋值：将 zeros(...) 的结果保存到 Muti_Cost
     for i=1:CrossPointNo  % 详解: for 循环：迭代变量 i 遍历 1:CrossPointNo
         [a b]=ShortestPath_Djk(Cost,CrossPointNo,i);  % 详解: 执行语句
         Muti_Cost(i,:)=a  % 详解: 执行语句
     end  % 详解: 执行语句




