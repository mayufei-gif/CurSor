% 文件: estimationLog.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [PI_new A_new B_new] = estimationLog(q, xi, chi, gamma_in, gamma_out, seq, B, alph)  % 详解: 函数定义：estimationLog(q, xi, chi, gamma_in, gamma_out, seq, B, alph), 返回：PI_new A_new B_new


[prodY prodX] = find(q(:,:,1)==1 & q(:,:,2)==0);  % 详解: 执行语句
[allY allX] = find(q(:,:,1)==1);  % 详解: 执行语句
PI_new = -Inf(size(q,2),size(q,2),size(q,1)-1);  % 详解: 赋值：计算表达式并保存到 PI_new
A_new = -Inf(size(q,2),size(q,2),size(q,1)-1);  % 详解: 赋值：计算表达式并保存到 A_new
B_new = -Inf(size(B));  % 详解: 赋值：计算表达式并保存到 B_new


sArray = findSArray(q,1,1);  % 详解: 赋值：将 findSArray(...) 的结果保存到 sArray
for i = sArray(1):sArray(end-1)  % 详解: for 循环：迭代变量 i 遍历 sArray(1):sArray(end-1)
    summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
    for k = sArray(1):sArray(end-1)  % 详解: for 循环：迭代变量 k 遍历 sArray(1):sArray(end-1)
        summ = logSum(summ, chi(1,2,k));  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
    end  % 详解: 执行语句
    PI_new(1,i,1)  = logProd(chi(1,2,i),-summ);  % 详解: 调用函数：PI_new(1,i,1) = logProd(chi(1,2,i),-summ)
end  % 详解: 执行语句
for d = 3:max(allY)  % 详解: for 循环：迭代变量 d 遍历 3:max(allY)
    for i = 1:max(allX)  % 详解: for 循环：迭代变量 i 遍历 1:max(allX)
        if q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,1)==1)
            parent = find(cumsum(q(d-1,:,2))>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 parent
            jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
            jArray = jArray(1):jArray(end-1);  % 详解: 赋值：将 jArray(...) 的结果保存到 jArray
            summOben = NaN;  % 详解: 赋值：计算表达式并保存到 summOben
            for t = 1:length(seq)  % 详解: for 循环：迭代变量 t 遍历 1:length(seq)
                summOben = logSum(summOben, chi(t,d,i));  % 详解: 赋值：将 logSum(...) 的结果保存到 summOben
            end  % 详解: 执行语句
            summUnten = NaN;  % 详解: 赋值：计算表达式并保存到 summUnten
            for m = jArray  % 详解: for 循环：迭代变量 m 遍历 jArray
                summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                for t = 1:length(seq)  % 详解: for 循环：迭代变量 t 遍历 1:length(seq)
                    summ = logSum(summ, chi(t,d,m));  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                end  % 详解: 执行语句
                summUnten = logSum(summUnten, summ);  % 详解: 赋值：将 logSum(...) 的结果保存到 summUnten
            end  % 详解: 执行语句
            PI_new(parent,i,d-1) = logProd(summOben, -summUnten);  % 详解: 调用函数：PI_new(parent,i,d-1) = logProd(summOben, -summUnten)
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

for d = 2:max(allY)  % 详解: for 循环：迭代变量 d 遍历 2:max(allY)
    for i = 1:max(allX)  % 详解: for 循环：迭代变量 i 遍历 1:max(allX)
        if q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,1)==1)
            jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
            for j = jArray  % 详解: for 循环：迭代变量 j 遍历 jArray
                oben = NaN;  % 详解: 赋值：计算表达式并保存到 oben
                for t = 1:length(seq)  % 详解: for 循环：迭代变量 t 遍历 1:length(seq)
                    oben = logSum(oben, xi(t,d,i,d,j));  % 详解: 赋值：将 logSum(...) 的结果保存到 oben
                end  % 详解: 执行语句
                
                unten = NaN;  % 详解: 赋值：计算表达式并保存到 unten
                for t = 1:length(seq)  % 详解: for 循环：迭代变量 t 遍历 1:length(seq)
                    unten = logSum(unten, gamma_out(t,d,i));  % 详解: 赋值：将 logSum(...) 的结果保存到 unten
                end  % 详解: 执行语句
                A_new(i,j,d-1) = logProd(oben,-unten);  % 详解: 调用函数：A_new(i,j,d-1) = logProd(oben,-unten)
                if isinf(A_new(i,j,d-1))  % 详解: 条件判断：if (isinf(A_new(i,j,d-1)))
                    A_new(i,j,d-1) = log(eps);  % 详解: 调用函数：A_new(i,j,d-1) = log(eps)
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

for v = alph  % 详解: for 循环：迭代变量 v 遍历 alph
    for d = 2:max(prodY)  % 详解: for 循环：迭代变量 d 遍历 2:max(prodY)
        for i = 1:max(prodX)  % 详解: for 循环：迭代变量 i 遍历 1:max(prodX)
            if q(d,i,2)==0 && q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,2)==0 && q(d,i,1)==1)
                d1 = NaN;  % 详解: 赋值：计算表达式并保存到 d1
                for n = 1:length(seq)  % 详解: for 循环：迭代变量 n 遍历 1:length(seq)
                    if seq(n)==v  % 详解: 条件判断：if (seq(n)==v)
                        d1 = logSum(d1, chi(n,d,i));  % 详解: 赋值：将 logSum(...) 的结果保存到 d1
                    end  % 详解: 执行语句
                end  % 详解: 执行语句
                d2 = NaN;  % 详解: 赋值：计算表达式并保存到 d2
                for n = 2:length(seq)  % 详解: for 循环：迭代变量 n 遍历 2:length(seq)
                    if seq(n)==v  % 详解: 条件判断：if (seq(n)==v)
                        d2 = logSum(d2, gamma_in(n,d,i));  % 详解: 赋值：将 logSum(...) 的结果保存到 d2
                    end  % 详解: 执行语句
                end  % 详解: 执行语句

                d3 = NaN;  % 详解: 赋值：计算表达式并保存到 d3
                for n = 1:length(seq)  % 详解: for 循环：迭代变量 n 遍历 1:length(seq)
                    d3 = logSum(d3, chi(n,d,i));  % 详解: 赋值：将 logSum(...) 的结果保存到 d3
                end  % 详解: 执行语句
                d4 = NaN;  % 详解: 赋值：计算表达式并保存到 d4
                for n = 2:length(seq)  % 详解: for 循环：迭代变量 n 遍历 2:length(seq)
                    d4 = logSum(d4, gamma_in(n,d,i));  % 详解: 赋值：将 logSum(...) 的结果保存到 d4
                end  % 详解: 执行语句
                x1 = logSum(d1,d2);  % 详解: 赋值：将 logSum(...) 的结果保存到 x1
                x2 = logSum(d3,d4);  % 详解: 赋值：将 logSum(...) 的结果保存到 x2
                B_new(d,i,v) = logProd(x1,-x2);  % 详解: 调用函数：B_new(d,i,v) = logProd(x1,-x2)
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

A_new = exp(A_new);  % 详解: 赋值：将 exp(...) 的结果保存到 A_new
PI_new = exp(PI_new);  % 详解: 赋值：将 exp(...) 的结果保存到 PI_new
B_new = exp(B_new);  % 详解: 赋值：将 exp(...) 的结果保存到 B_new


end  % 详解: 执行语句




