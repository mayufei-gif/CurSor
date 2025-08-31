% 文件: expectationAlphaBetaLog.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [ alpha, beta ] = expectationAlphaBetaLog( a, pi, q, b, seq )  % 详解: 函数定义：expectationAlphaBetaLog(a, pi, q, b, seq), 返回： alpha, beta 

[prodY prodX] = find(q(:,:,1)==1 & q(:,:,2)==0);  % 详解: 执行语句
[intY intX] = find(q(:,:,2)~=0);  % 详解: 执行语句

alpha = -Inf(length(seq),length(seq),size(q,1),size(q,2));  % 详解: 赋值：计算表达式并保存到 alpha
 beta = -Inf(length(seq),length(seq),size(q,1),size(q,2));  % 详解: 赋值：计算表达式并保存到 beta


for t = length(seq):-1:1  % 详解: for 循环：迭代变量 t 遍历 length(seq):-1:1
    for k = 0:(length(seq)-t)  % 详解: for 循环：迭代变量 k 遍历 0:(length(seq)-t)
        for d = max(prodY):-1:2  % 详解: for 循环：迭代变量 d 遍历 max(prodY):-1:2
            for i = 1:max(prodX)  % 详解: for 循环：迭代变量 i 遍历 1:max(prodX)
                if q(d,i,2)==0 && q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,2)==0 && q(d,i,1)==1)
                    parent = find(cumsum(q(d-1,:,2))>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 parent
                    jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
                    if k==0  % 详解: 条件判断：if (k==0)
                        alpha(t,t,d,i) = logProd(pi(parent,i,d-1), b(d,i,seq(t)));  % 详解: 调用函数：alpha(t,t,d,i) = logProd(pi(parent,i,d-1), b(d,i,seq(t)))
                    else  % 详解: 条件判断：else 分支
                        summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                        for j = jArray(1:end-1)  % 详解: for 循环：迭代变量 j 遍历 jArray(1:end-1)
                            pro = logProd(alpha(t,t+k-1,d,j), a(j,i,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                            summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                        end  % 详解: 执行语句
                        alpha(t,t+k,d,i) = logProd(summ, b(d,i,seq(t+k)));  % 详解: 调用函数：alpha(t,t+k,d,i) = logProd(summ, b(d,i,seq(t+k)))
                    end  % 详解: 执行语句
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        
        for d = max(intY):-1:2  % 详解: for 循环：迭代变量 d 遍历 max(intY):-1:2
            for i = 1:max(intX)  % 详解: for 循环：迭代变量 i 遍历 1:max(intX)
                if q(d,i,2)~=0  % 详解: 条件判断：if (q(d,i,2)~=0)
                    parent = find(cumsum(q(d-1,:,2))>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 parent
                    jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
                    sArray = findSArray(q,d,i);  % 详解: 赋值：将 findSArray(...) 的结果保存到 sArray
                    if k==0  % 详解: 条件判断：if (k==0)
                        summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                        for s = sArray(1:end-1)  % 详解: for 循环：迭代变量 s 遍历 sArray(1:end-1)
                            pro = logProd(alpha(t,t,d+1,s), a(s,sArray(end),d));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                            summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                        end  % 详解: 执行语句
                        alpha(t,t,d,i) = logProd(summ, pi(parent,i,d-1));  % 详解: 调用函数：alpha(t,t,d,i) = logProd(summ, pi(parent,i,d-1))
                    else  % 详解: 条件判断：else 分支
                        outerSumm = NaN;  % 详解: 赋值：计算表达式并保存到 outerSumm
                        for l = 0:(k-1)  % 详解: for 循环：迭代变量 l 遍历 0:(k-1)
                            summ1 = NaN;  % 详解: 赋值：计算表达式并保存到 summ1
                            for j = jArray(1:end-1)  % 详解: for 循环：迭代变量 j 遍历 jArray(1:end-1)
                                pro1 = logProd(alpha(t,t+l,d,j), a(j,i,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro1
                                summ1 = logSum(summ1,pro1);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ1
                            end  % 详解: 执行语句

                            summ2 = NaN;  % 详解: 赋值：计算表达式并保存到 summ2
                            for s = sArray(1:end-1)  % 详解: for 循环：迭代变量 s 遍历 sArray(1:end-1)
                                pro2 = logProd(alpha(t+l+1,t+k,d+1,s), a(s,sArray(end),d));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro2
                                summ2 = logSum(summ2,pro2);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ2
                            end  % 详解: 执行语句
                            outerSumm = logSum(outerSumm,logProd(summ1,summ2));  % 详解: 赋值：将 logSum(...) 的结果保存到 outerSumm
                        end  % 详解: 执行语句

                        summ3 = NaN;  % 详解: 赋值：计算表达式并保存到 summ3
                        for s = sArray(1:end-1);  % 详解: for 循环：迭代变量 s 遍历 sArray(1:end-1);
                            pro3 = logProd(alpha(t,t+k,d+1,s), a(s,sArray(end),d));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro3
                            summ3 = logSum(summ3,pro3);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ3
                        end  % 详解: 执行语句
                        alpha(t,t+k,d,i) = logSum(outerSumm,logProd(summ3,pi(parent,i,d-1)));  % 详解: 调用函数：alpha(t,t+k,d,i) = logSum(outerSumm,logProd(summ3,pi(parent,i,d-1)))
                    end  % 详解: 执行语句
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句

        for d = max(prodY):-1:2  % 详解: for 循环：迭代变量 d 遍历 max(prodY):-1:2
            for i = 1:max(prodX)  % 详解: for 循环：迭代变量 i 遍历 1:max(prodX)
                if q(d,i,2)==0 && q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,2)==0 && q(d,i,1)==1)
                    jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
                    jArray = jArray(1:end-1);  % 详解: 赋值：将 jArray(...) 的结果保存到 jArray
                    e = find(q(d,i:end,1)==2,1) + i-1;  % 详解: 赋值：将 find(...) 的结果保存到 e
                    if k==0  % 详解: 条件判断：if (k==0)
                        beta(t,t,d,i) = logProd(b(d,i,seq(t)),a(i,e,d-1));  % 详解: 调用函数：beta(t,t,d,i) = logProd(b(d,i,seq(t)),a(i,e,d-1))
                    else  % 详解: 条件判断：else 分支
                        summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                        for j = jArray  % 详解: for 循环：迭代变量 j 遍历 jArray
                            pro = logProd(beta(t+1,t+k,d,j),a(i,j,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                            summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                        end  % 详解: 执行语句
                        beta(t,t+k,d,i) = logProd(summ,b(d,i,seq(t)));  % 详解: 调用函数：beta(t,t+k,d,i) = logProd(summ,b(d,i,seq(t)))
                    end  % 详解: 执行语句
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        
        for d = max(intY):-1:2  % 详解: for 循环：迭代变量 d 遍历 max(intY):-1:2
            for i = 1:max(intX)  % 详解: for 循环：迭代变量 i 遍历 1:max(intX)
                if q(d,i,2)~=0  % 详解: 条件判断：if (q(d,i,2)~=0)
                    jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
                    sArray = findSArray(q,d,i);  % 详解: 赋值：将 findSArray(...) 的结果保存到 sArray
                    if k==0  % 详解: 条件判断：if (k==0)
                        summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                        for s = sArray(1:end-1)  % 详解: for 循环：迭代变量 s 遍历 sArray(1:end-1)
                            pro = logProd(beta(t,t,d+1,s), pi(i,s,d));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                            summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                        end  % 详解: 执行语句
                        beta(t,t,d,i) = logProd(summ, a(i,jArray(end),d-1));  % 详解: 调用函数：beta(t,t,d,i) = logProd(summ, a(i,jArray(end),d-1))
                    else  % 详解: 条件判断：else 分支
                        outerSumm = NaN;  % 详解: 赋值：计算表达式并保存到 outerSumm
                        for i1 = 0:k-1  % 详解: for 循环：迭代变量 i1 遍历 0:k-1
                            summ1 = NaN;  % 详解: 赋值：计算表达式并保存到 summ1
                            for s = sArray(1:end-1)  % 详解: for 循环：迭代变量 s 遍历 sArray(1:end-1)
                                pro1 = logProd(beta(t,t+i1,d+1,s),pi(i,s,d));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro1
                                summ1 = logSum(summ1,pro1);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ1
                            end  % 详解: 执行语句

                            summ2 = NaN;  % 详解: 赋值：计算表达式并保存到 summ2
                            for j = jArray(1:end-1)  % 详解: for 循环：迭代变量 j 遍历 jArray(1:end-1)
                                pro2 = logProd(beta(t+i1+1,t+k,d,j),a(i,j,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro2
                                summ2 = logSum(summ2,pro2);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ2
                            end  % 详解: 执行语句
                            outerSumm = logSum(outerSumm,logProd(summ1,summ2));  % 详解: 赋值：将 logSum(...) 的结果保存到 outerSumm
                        end  % 详解: 执行语句

                        summ3 = NaN;  % 详解: 赋值：计算表达式并保存到 summ3
                        for s = sArray(1:end-1)  % 详解: for 循环：迭代变量 s 遍历 sArray(1:end-1)
                            pro3 = logProd(beta(t,t+k,d+1,s), pi(i,s,d));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro3
                            summ3 = logSum(summ3,pro3);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ3
                        end  % 详解: 执行语句
                        beta(t,t+k,d,i) = logSum(outerSumm,logProd(summ3, a(i,jArray(end),d-1)));  % 详解: 调用函数：beta(t,t+k,d,i) = logSum(outerSumm,logProd(summ3, a(i,jArray(end),d-1)))
                    end  % 详解: 执行语句
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

end  % 详解: 执行语句



