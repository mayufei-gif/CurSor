% 文件: optimalMatchingTest.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% Consider matching sources to detections

%  s1 d2  
%         s2 d3
%  d1

a  = optimalMatching([52;0.01])  % 详解: 赋值：将 optimalMatching(...) 的结果保存到 a

sources = [0.1 0.7; 0.6 0.4]';  % 赋值：设置变量 sources  % 详解: 赋值：计算表达式并保存到 sources  % 详解: 赋值：计算表达式并保存到 sources
detections = [0.2 0.2; 0.2 0.8; 0.7 0.1]';  % 赋值：设置变量 detections  % 详解: 赋值：计算表达式并保存到 detections  % 详解: 赋值：计算表达式并保存到 detections
dst = sqdist(sources, detections)  % 详解: 赋值：将 sqdist(...) 的结果保存到 dst

a = optimalMatching(dst)  % 详解: 赋值：将 optimalMatching(...) 的结果保存到 a

a = optimalMatching(dst')  % 赋值：设置变量 a  % 详解: 赋值：将 optimalMatching(...) 的结果保存到 a  % 详解: 赋值：将 optimalMatching(...) 的结果保存到 a




