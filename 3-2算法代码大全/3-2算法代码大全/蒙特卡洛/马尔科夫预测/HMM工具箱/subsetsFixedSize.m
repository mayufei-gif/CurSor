% 文件: subsetsFixedSize.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function sub_s=subsets1(s,k)  % 详解: 执行语句

if k<0  % 详解: 条件判断：if (k<0)
    error('subset size must be positive');  % 详解: 调用函数：error('subset size must be positive')
elseif k==0  % 详解: 条件判断：elseif (k==0)
    sub_s={[]};  % 详解: 赋值：计算表达式并保存到 sub_s
else  % 详解: 条件判断：else 分支
    l=length(s);  % 详解: 赋值：将 length(...) 的结果保存到 l
    ss={};  % 详解: 赋值：计算表达式并保存到 ss
    if l>=k  % 详解: 条件判断：if (l>=k)
        if k==1  % 详解: 条件判断：if (k==1)
            for I=1:l  % 详解: for 循环：迭代变量 I 遍历 1:l
                ss{I}=s(I);  % 详解: 执行语句
            end  % 详解: 执行语句
        else  % 详解: 条件判断：else 分支
            for I=1:l  % 详解: for 循环：迭代变量 I 遍历 1:l
                ss1=subsets1(s([(I+1):l]),k-1);  % 详解: 赋值：将 subsets1(...) 的结果保存到 ss1
                for J=1:length(ss1)  % 详解: for 循环：迭代变量 J 遍历 1:length(ss1)
                    ss{end+1}=[s(I),ss1{J}];  % 详解: 执行语句
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    sub_s=ss;  % 详解: 赋值：计算表达式并保存到 sub_s
end  % 详解: 执行语句




