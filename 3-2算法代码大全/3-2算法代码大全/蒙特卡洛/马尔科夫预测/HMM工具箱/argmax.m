% 文件: argmax.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function indices = argmax(v)  % 详解: 执行语句

[m i] = max(v(:));  % 详解: 统计：最大/最小值
indices = ind2subv(mysize(v), i);  % 详解: 赋值：将 ind2subv(...) 的结果保存到 indices




