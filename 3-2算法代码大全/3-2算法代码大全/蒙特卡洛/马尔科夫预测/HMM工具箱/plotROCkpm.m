% 文件: plotROCkpm.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [falseAlarmRate, detectionRate, area, th] = plotROC(confidence, testClass, col, varargin)  % 详解: 函数定义：plotROC(confidence, testClass, col, varargin), 返回：falseAlarmRate, detectionRate, area, th

if nargin < 3, col = []; end  % 详解: 条件判断：if (nargin < 3, col = []; end)
[scale01] = process_options(varargin, 'scale01', 1);  % 详解: 执行语句

S = rand('state');  % 详解: 赋值：将 rand(...) 的结果保存到 S
rand('state',0);  % 详解: 调用函数：rand('state',0)
confidence = confidence + rand(size(confidence))*10^(-10);  % 详解: 赋值：计算表达式并保存到 confidence
rand('state',S)  % 详解: 调用函数：rand('state',S)

ndxAbs = find(testClass==0);  % 详解: 赋值：将 find(...) 的结果保存到 ndxAbs
ndxPres = find(testClass==1);  % 详解: 赋值：将 find(...) 的结果保存到 ndxPres

[th, j] = sort(confidence(ndxAbs));  % 详解: 执行语句
th = th(fix(linspace(1, length(th), 1250)));  % 详解: 赋值：将 th(...) 的结果保存到 th

cAbs = confidence(ndxAbs);  % 详解: 赋值：将 confidence(...) 的结果保存到 cAbs
cPres = confidence(ndxPres);  % 详解: 赋值：将 confidence(...) 的结果保存到 cPres
for t=1:length(th)  % 详解: for 循环：迭代变量 t 遍历 1:length(th)
  if length(ndxPres) == 0  % 详解: 条件判断：if (length(ndxPres) == 0)
    detectionRate(t) = 0;  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    detectionRate(t)  = sum(cPres>=th(t)) / length(ndxPres);  % 详解: 调用函数：detectionRate(t) = sum(cPres>=th(t)) / length(ndxPres)
  end  % 详解: 执行语句
  if length(ndxAbs) == 0  % 详解: 条件判断：if (length(ndxAbs) == 0)
    falseAlarmRate(t) = 0;  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    falseAlarmRate(t) = sum(cAbs>=th(t)) / length(ndxAbs);  % 详解: 调用函数：falseAlarmRate(t) = sum(cAbs>=th(t)) / length(ndxAbs)
  end  % 详解: 执行语句
  
end  % 详解: 执行语句

area = sum(abs(falseAlarmRate(2:end) - falseAlarmRate(1:end-1)) .* detectionRate(2:end));  % 详解: 赋值：将 sum(...) 的结果保存到 area

if ~isempty(col)  % 详解: 条件判断：if (~isempty(col))
    h=plot(falseAlarmRate, detectionRate, [col '-']);  % 详解: 赋值：将 plot(...) 的结果保存到 h
    e = 0.05;  % 详解: 赋值：计算表达式并保存到 e
    if scale01  % 详解: 条件判断：if (scale01)
      axis([0-e 1+e 0-e 1+e])  % 详解: 调用函数：axis([0-e 1+e 0-e 1+e])
    else  % 详解: 条件判断：else 分支
      axis([0-e 0.5+e 0.5-e 1+e])  % 详解: 调用函数：axis([0-e 0.5+e 0.5-e 1+e])
    end  % 详解: 执行语句
    grid on  % 详解: 执行语句
    ylabel('detection rate')  % 详解: 调用函数：ylabel('detection rate')
    xlabel('false alarm rate')  % 详解: 调用函数：xlabel('false alarm rate')
end  % 详解: 执行语句




