% 文件: hsvKPM.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function colors = hsvKPM(N)  % 详解: 执行语句

colors = hsv(N);  % 详解: 赋值：将 hsv(...) 的结果保存到 colors
perm = randperm(N);  % 详解: 赋值：将 randperm(...) 的结果保存到 perm
colors = colors(perm,:);  % 详解: 赋值：将 colors(...) 的结果保存到 colors




