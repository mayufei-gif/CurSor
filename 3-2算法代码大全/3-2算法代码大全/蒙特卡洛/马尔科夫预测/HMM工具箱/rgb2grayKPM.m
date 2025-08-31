% 文件: rgb2grayKPM.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function g = rgb2grayKPM(rgb)  % 详解: 执行语句

[nr nc ncolors] = size(rgb);  % 详解: 获取向量/矩阵尺寸
if ncolors > 1  % 详解: 条件判断：if (ncolors > 1)
  g = rgb2gray(rgb);  % 详解: 赋值：将 rgb2gray(...) 的结果保存到 g
else  % 详解: 条件判断：else 分支
  g = rgb;  % 详解: 赋值：计算表达式并保存到 g
end  % 详解: 执行语句





