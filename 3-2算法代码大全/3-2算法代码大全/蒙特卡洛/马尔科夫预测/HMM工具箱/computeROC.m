% 文件: computeROC.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [FPrate, TPrate, AUC, thresholds] = computeROC(confidence, testClass)  % 详解: 函数定义：computeROC(confidence, testClass), 返回：FPrate, TPrate, AUC, thresholds

S = rand('state');  % 详解: 赋值：将 rand(...) 的结果保存到 S
rand('state',0);  % 详解: 调用函数：rand('state',0)
confidence = confidence + rand(size(confidence))*10^(-10);  % 详解: 赋值：计算表达式并保存到 confidence
rand('state',S)  % 详解: 调用函数：rand('state',S)
[thresholds order] = sort(confidence, 'descend');  % 详解: 执行语句
testClass = testClass(order);  % 详解: 赋值：将 testClass(...) 的结果保存到 testClass

AUC = 0;  % 详解: 赋值：计算表达式并保存到 AUC
faCnt = 0;  % 详解: 赋值：计算表达式并保存到 faCnt
tpCnt = 0;  % 详解: 赋值：计算表达式并保存到 tpCnt
falseAlarms = zeros(1,size(thresholds,2));  % 详解: 赋值：将 zeros(...) 的结果保存到 falseAlarms
detections = zeros(1,size(thresholds,2));  % 详解: 赋值：将 zeros(...) 的结果保存到 detections
fPrev = -inf;  % 详解: 赋值：计算表达式并保存到 fPrev
faPrev = 0;  % 详解: 赋值：计算表达式并保存到 faPrev
tpPrev = 0;  % 详解: 赋值：计算表达式并保存到 tpPrev

P = max(size(find(testClass==1)));  % 详解: 赋值：将 max(...) 的结果保存到 P
N = max(size(find(testClass==0)));  % 详解: 赋值：将 max(...) 的结果保存到 N

for i=1:length(thresholds)  % 详解: for 循环：迭代变量 i 遍历 1:length(thresholds)
    if thresholds(i) ~= fPrev  % 详解: 条件判断：if (thresholds(i) ~= fPrev)
        falseAlarms(i) = faCnt;  % 详解: 执行语句
        detections(i) = tpCnt;  % 详解: 执行语句

        AUC = AUC + polyarea([faPrev faPrev faCnt/N faCnt/N],[0 tpPrev tpCnt/P 0]);  % 详解: 赋值：计算表达式并保存到 AUC

        fPrev = thresholds(i);  % 详解: 赋值：将 thresholds(...) 的结果保存到 fPrev
        faPrev = faCnt/N;  % 详解: 赋值：计算表达式并保存到 faPrev
        tpPrev = tpCnt/P;  % 详解: 赋值：计算表达式并保存到 tpPrev
    end  % 详解: 执行语句
    
    if testClass(i) == 1  % 详解: 条件判断：if (testClass(i) == 1)
        tpCnt = tpCnt + 1;  % 详解: 赋值：计算表达式并保存到 tpCnt
    else  % 详解: 条件判断：else 分支
        faCnt = faCnt + 1;  % 详解: 赋值：计算表达式并保存到 faCnt
    end  % 详解: 执行语句
end  % 详解: 执行语句

AUC = AUC + polyarea([faPrev faPrev 1 1],[0 tpPrev 1 0]);  % 详解: 赋值：计算表达式并保存到 AUC

FPrate = falseAlarms/N;  % 详解: 赋值：计算表达式并保存到 FPrate
TPrate = detections/P;  % 详解: 赋值：计算表达式并保存到 TPrate




