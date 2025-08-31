% 文件: VQIndex.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [I, dst]=VQIndex(X,CB)  % 详解: 函数定义：VQIndex(X,CB), 返回：I, dst

L=size(CB,2);  % 详解: 赋值：将 size(...) 的结果保存到 L
N=size(X,2);  % 详解: 赋值：将 size(...) 的结果保存到 N
LNThreshold=64*10000;  % 详解: 赋值：计算表达式并保存到 LNThreshold

if L*N<LNThreshold  % 详解: 条件判断：if (L*N<LNThreshold)
    D=zeros(L,N);  % 详解: 赋值：将 zeros(...) 的结果保存到 D
    for i=1:L  % 详解: for 循环：迭代变量 i 遍历 1:L
        D(i,:)=sum((repmat(CB(:,i),1,N)-X).^2,1);  % 详解: 调用函数：D(i,:)=sum((repmat(CB(:,i),1,N)-X).^2,1)
    end  % 详解: 执行语句
    [dst I]=min(D);  % 详解: 统计：最大/最小值
else  % 详解: 条件判断：else 分支
    I=zeros(1,N);  % 详解: 赋值：将 zeros(...) 的结果保存到 I
    dst=I;  % 详解: 赋值：计算表达式并保存到 dst
    for i=1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
        D=sum((repmat(X(:,i),1,L)-CB).^2,1);  % 详解: 赋值：将 sum(...) 的结果保存到 D
        [dst(i) I(i)]=min(D);  % 详解: 统计：最大/最小值
    end  % 详解: 执行语句
end  % 详解: 执行语句




