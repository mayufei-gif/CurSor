% 文件: expectationEtaLog.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [ eta_in eta_out ] = expectationEtaLog( a, pi, q, alpha, beta, seq )  % 详解: 函数定义：expectationEtaLog(a, pi, q, alpha, beta, seq), 返回： eta_in eta_out 

[allY allX] = find(q(:,:,1)==1);  % 详解: 执行语句

eta_in = -Inf(length(seq),size(q,1),size(q,2));  % 详解: 赋值：计算表达式并保存到 eta_in
eta_out = -Inf(length(seq),size(q,1),size(q,2));  % 详解: 赋值：计算表达式并保存到 eta_out

sArray = findSArray(q,1,1);  % 详解: 赋值：将 findSArray(...) 的结果保存到 sArray
for i = sArray(1):sArray(end-1)  % 详解: for 循环：迭代变量 i 遍历 sArray(1):sArray(end-1)
    for t = 1:length(seq)  % 详解: for 循环：迭代变量 t 遍历 1:length(seq)
        if t==1  % 详解: 条件判断：if (t==1)
            eta_in(t,2,i) = pi(1,i,1);  % 详解: 调用函数：eta_in(t,2,i) = pi(1,i,1)
        else  % 详解: 条件判断：else 分支
            summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
            jArray = findJArray(q,2,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
            for j = jArray  % 详解: for 循环：迭代变量 j 遍历 jArray
                pro = logProd(alpha(1,t-1,2,j),a(j,i,1));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
            end  % 详解: 执行语句
            eta_in(t,2,i) = summ;  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句


for d = 3:max(allY)  % 详解: for 循环：迭代变量 d 遍历 3:max(allY)
	for i = 1:max(allX)  % 详解: for 循环：迭代变量 i 遍历 1:max(allX)
        if q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,1)==1)
            parent = find(cumsum(q(d-1,:,2))>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 parent
            jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
            for t = 1:length(seq)  % 详解: for 循环：迭代变量 t 遍历 1:length(seq)
                if t==1  % 详解: 条件判断：if (t==1)
                    eta_in(t,d,i) = logProd(eta_in(t,d-1,parent),pi(parent,i,d-1));  % 详解: 调用函数：eta_in(t,d,i) = logProd(eta_in(t,d-1,parent),pi(parent,i,d-1))
                else  % 详解: 条件判断：else 分支
                    outerSumm = NaN;  % 详解: 赋值：计算表达式并保存到 outerSumm
                    for ttick = 1:(t-1)  % 详解: for 循环：迭代变量 ttick 遍历 1:(t-1)
                        summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                        for j = jArray  % 详解: for 循环：迭代变量 j 遍历 jArray
                            pro = logProd(alpha(ttick,t-1,d,j),a(j,i,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                            summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                        end  % 详解: 执行语句
                        outerSumm = logSum(outerSumm,logProd(summ,eta_in(ttick,d-1,parent)));  % 详解: 赋值：将 logSum(...) 的结果保存到 outerSumm
                    end  % 详解: 执行语句
                    eta_in(t,d,i) = logSum(outerSumm,logProd(eta_in(t,d-1,parent),pi(parent,i,d-1)));  % 详解: 调用函数：eta_in(t,d,i) = logSum(outerSumm,logProd(eta_in(t,d-1,parent),pi(parent,i,d-1)))
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

sArray = findSArray(q,1,1);  % 详解: 赋值：将 findSArray(...) 的结果保存到 sArray
for i = sArray(1):sArray(end-1)  % 详解: for 循环：迭代变量 i 遍历 sArray(1):sArray(end-1)
    for t = length(seq):-1:1  % 详解: for 循环：迭代变量 t 遍历 length(seq):-1:1
        if t==length(seq)  % 详解: 条件判断：if (t==length(seq))
            ende = sArray(end);  % 详解: 赋值：将 sArray(...) 的结果保存到 ende
            eta_out(t,2,i) = a(i,ende,1);  % 详解: 调用函数：eta_out(t,2,i) = a(i,ende,1)
        else  % 详解: 条件判断：else 分支
            summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
            jArray = findJArray(q,2,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
            for j = jArray  % 详解: for 循环：迭代变量 j 遍历 jArray
                prod = logProd(beta(t+1,length(seq),2,j),a(i,j,1));  % 详解: 赋值：将 logProd(...) 的结果保存到 prod
                summ = logSum(summ,prod);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
            end  % 详解: 执行语句
            eta_out(t,2,i) = summ;  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

for d = 3:max(allY)  % 详解: for 循环：迭代变量 d 遍历 3:max(allY)
	for i = 1:max(allX)  % 详解: for 循环：迭代变量 i 遍历 1:max(allX)
        if q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,1)==1)
            parent = find(cumsum(q(d-1,:,2))>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 parent
            jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
            e = find(q(d,i:end,1)==2,1) + i-1;  % 详解: 赋值：将 find(...) 的结果保存到 e
            for t = length(seq):-1:1  % 详解: for 循环：迭代变量 t 遍历 length(seq):-1:1
                if t==length(seq)  % 详解: 条件判断：if (t==length(seq))
                    eta_out(t,d,i) = logProd(eta_out(t,d-1,parent),a(i,e,d-1));  % 详解: 调用函数：eta_out(t,d,i) = logProd(eta_out(t,d-1,parent),a(i,e,d-1))
                else  % 详解: 条件判断：else 分支
                    outerSumm = NaN;  % 详解: 赋值：计算表达式并保存到 outerSumm
                    for k = (t+1):length(seq)  % 详解: for 循环：迭代变量 k 遍历 (t+1):length(seq)
                        summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                        for j = jArray  % 详解: for 循环：迭代变量 j 遍历 jArray
                            pro = logProd(beta(t+1,k,d,j),a(i,j,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                            summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                        end  % 详解: 执行语句
                        outerSumm = logSum(outerSumm,logProd(summ,eta_out(k,d-1,parent)));  % 详解: 赋值：将 logSum(...) 的结果保存到 outerSumm
                    end  % 详解: 执行语句
                    eta_out(t,d,i) = logSum(outerSumm,logProd(eta_out(t,d-1,parent),a(i,e,d-1)));  % 详解: 调用函数：eta_out(t,d,i) = logSum(outerSumm,logProd(eta_out(t,d-1,parent),a(i,e,d-1)))
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句


end  % 详解: 执行语句




