% 文件: plotROC.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [falseAlarmRate, detectionRate, area, th] = plotROC(confidence, testClass, col, varargin)  % 详解: 函数定义：plotROC(confidence, testClass, col, varargin), 返回：falseAlarmRate, detectionRate, area, th

if nargin < 3, col = []; end  % 详解: 条件判断：if (nargin < 3, col = []; end)

[scale01] = process_options(varargin, 'scale01', 1);  % 详解: 执行语句

[falseAlarmRate detectionRate area th] = computeROC(confidence, testClass);  % 详解: 执行语句

if ~isempty(col)  % 详解: 条件判断：if (~isempty(col))
    h=plot(falseAlarmRate, detectionRate, [col '-']);  % 详解: 赋值：将 plot(...) 的结果保存到 h
    ex = 0.05*max(falseAlarmRate);  % 详解: 赋值：计算表达式并保存到 ex
    ey = 0.05;  % 详解: 赋值：计算表达式并保存到 ey
    if scale01  % 详解: 条件判断：if (scale01)
      axis([0-ex max(falseAlarmRate)+ex 0-ey 1+ey])  % 详解: 调用函数：axis([0-ex max(falseAlarmRate)+ex 0-ey 1+ey])
    else  % 详解: 条件判断：else 分支
      axis([0-ex max(falseAlarmRate)*0.5+ex 0.5-ey 1+ey])  % 详解: 调用函数：axis([0-ex max(falseAlarmRate)*0.5+ex 0.5-ey 1+ey])
    end  % 详解: 执行语句
    grid on  % 详解: 执行语句
    ylabel('detection rate')  % 详解: 调用函数：ylabel('detection rate')
    xlabel('false alarm rate')  % 详解: 调用函数：xlabel('false alarm rate')
end  % 详解: 执行语句





