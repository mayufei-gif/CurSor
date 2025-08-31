% 文件: assignEdgeNums.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [edge_id, nedges] = assignEdgeNums(adj_mat)  % 详解: 函数定义：assignEdgeNums(adj_mat), 返回：edge_id, nedges

nnodes = length(adj_mat);  % 详解: 赋值：将 length(...) 的结果保存到 nnodes
edge_id = zeros(nnodes);  % 详解: 赋值：将 zeros(...) 的结果保存到 edge_id
e = 1;  % 详解: 赋值：计算表达式并保存到 e
for i=1:nnodes  % 详解: for 循环：迭代变量 i 遍历 1:nnodes
  for j=i+1:nnodes  % 详解: for 循环：迭代变量 j 遍历 i+1:nnodes
    if adj_mat(i,j)  % 详解: 条件判断：if (adj_mat(i,j))
      edge_id(i,j) = e;  % 详解: 执行语句
      e = e+1;  % 详解: 赋值：计算表达式并保存到 e
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

nedges = e-1;  % 详解: 赋值：计算表达式并保存到 nedges
tmp = edge_id;  % 详解: 赋值：计算表达式并保存到 tmp
ndx = find(tmp);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
tmp(ndx) = tmp(ndx)+nedges;  % 详解: 执行语句
edge_id = edge_id + triu(tmp)';  % 赋值：设置变量 edge_id  % 详解: 赋值：计算表达式并保存到 edge_id  % 详解: 赋值：计算表达式并保存到 edge_id


if 0  % 详解: 条件判断：if (0)
ndx = find(adj_mat);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
nedges = length(ndx);  % 详解: 赋值：将 length(...) 的结果保存到 nedges
nnodes = length(adj_mat);  % 详解: 赋值：将 length(...) 的结果保存到 nnodes
edge_id = zeros(1, nnodes*nnodes);  % 详解: 赋值：将 zeros(...) 的结果保存到 edge_id
edge_id(ndx) = 1:nedges;  % 详解: 执行语句
edge_id = reshape(edge_id, nnodes, nnodes);  % 详解: 赋值：将 reshape(...) 的结果保存到 edge_id
end  % 详解: 执行语句




