% 文件: expectationXiChiLog.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [ xi chi gamma_in gamma_out ] = expectationXiChiLog(a, pi, q, eta_in, eta_out, alpha, beta, seq)  % 详解: 函数定义：expectationXiChiLog(a, pi, q, eta_in, eta_out, alpha, beta, seq), 返回： xi chi gamma_in gamma_out 

POlambda = log(sum(exp(alpha(1,length(seq),2,:,1,1))));  % 详解: 赋值：将 log(...) 的结果保存到 POlambda
[allY allX] = find(q(:,:,1)==1);  % 详解: 执行语句

xi = -Inf(length(seq),size(q,1),size(q,2)-1,size(q,1),size(q,2));  % 详解: 赋值：计算表达式并保存到 xi
chi = -Inf(length(seq),size(q,1),size(q,2)-1);  % 详解: 赋值：计算表达式并保存到 chi
gamma_in = chi;  % 详解: 赋值：计算表达式并保存到 gamma_in
gamma_out= chi;  % 详解: 赋值：计算表达式并保存到 gamma_out

for t = 1:length(seq);  % 详解: for 循环：迭代变量 t 遍历 1:length(seq);
    sArray = findSArray(q,1,1);  % 详解: 赋值：将 findSArray(...) 的结果保存到 sArray
    for i = sArray(1:end-1)  % 详解: for 循环：迭代变量 i 遍历 sArray(1:end-1)
        jArray = findJArray(q,2,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
        for k = jArray(1:end-1)  % 详解: for 循环：迭代变量 k 遍历 jArray(1:end-1)
            if t==length(seq)  % 详解: 条件判断：if (t==length(seq))
                temp = logProd(alpha(1,t,2,i),a(i,jArray(end),1));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                xi(t,2,i,2,jArray(end)) = logProd(temp, -POlambda);  % 详解: 调用函数：xi(t,2,i,2,jArray(end)) = logProd(temp, -POlambda)
            else  % 详解: 条件判断：else 分支
                temp = logProd(alpha(1,t,2,i),beta(t+1,length(seq),2,k));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                temp = logProd(temp,a(i,k,1));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                xi(t,2,i,2,k) = logProd(temp, -POlambda);  % 详解: 调用函数：xi(t,2,i,2,k) = logProd(temp, -POlambda)
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句


for t = 1:(length(seq))  % 详解: for 循环：迭代变量 t 遍历 1:(length(seq))
    for d = 3:max(allY)  % 详解: for 循环：迭代变量 d 遍历 3:max(allY)
    	for i = 1:max(allX)  % 详解: for 循环：迭代变量 i 遍历 1:max(allX)
            if q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,1)==1)
                parent = find(cumsum(q(d-1,:,2))>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 parent
                jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
                for j = jArray  % 详解: for 循环：迭代变量 j 遍历 jArray
                    if j == jArray(end)  % 详解: 条件判断：if (j == jArray(end))
                        summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                        for s = 1:t  % 详解: for 循环：迭代变量 s 遍历 1:t
                            pro = logProd(eta_in(s,d-1,parent),alpha(s,t,d,i));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                            summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                        end  % 详解: 执行语句
                        temp = logProd(summ, a(i,j,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                        temp = logProd(temp, eta_out(t,d-1,parent));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                        xi(t,d,i,d,j) = logProd(temp, -POlambda);  % 详解: 调用函数：xi(t,d,i,d,j) = logProd(temp, -POlambda)
                    else  % 详解: 条件判断：else 分支
                        if t ~= length(seq)  % 详解: 条件判断：if (t ~= length(seq))
                            summ1 = NaN;  % 详解: 赋值：计算表达式并保存到 summ1
                            for s = 1:t  % 详解: for 循环：迭代变量 s 遍历 1:t
                                pro1 = logProd(eta_in(s,d-1,parent), alpha(s,t,d,i));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro1
                                summ1 = logSum(summ1,pro1);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ1
                            end  % 详解: 执行语句

                            summ2 = NaN;  % 详解: 赋值：计算表达式并保存到 summ2
                            for e = (t+1):length(seq)  % 详解: for 循环：迭代变量 e 遍历 (t+1):length(seq)
                                pro2 = logProd(eta_out(e,d-1,parent), beta(t+1,e,d,j));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro2
                                summ2 = logSum(summ2,pro2);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ2
                            end  % 详解: 执行语句

                            temp = logProd(summ1,summ2);  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                            temp = logProd(temp, a(i,j,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                            xi(t,d,i,d,j) = logProd(temp, -POlambda);  % 详解: 调用函数：xi(t,d,i,d,j) = logProd(temp, -POlambda)
                        end  % 详解: 执行语句
                    end  % 详解: 执行语句
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句



sArray = findSArray(q,1,1);  % 详解: 赋值：将 findSArray(...) 的结果保存到 sArray
for i = sArray(1):sArray(end-1);  % 详解: for 循环：迭代变量 i 遍历 sArray(1):sArray(end-1);
    temp = logProd(pi(1,i,1), beta(1,length(seq),2,i));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
    chi(1,2,i) = logProd(temp, -POlambda);  % 详解: 调用函数：chi(1,2,i) = logProd(temp, -POlambda)
end  % 详解: 执行语句

for t=1:length(seq)  % 详解: for 循环：迭代变量 t 遍历 1:length(seq)
    for d = 3:max(allY)  % 详解: for 循环：迭代变量 d 遍历 3:max(allY)
    	for i = 1:max(allX)  % 详解: for 循环：迭代变量 i 遍历 1:max(allX)
            if q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,1)==1)
                parent = find(cumsum(q(d-1,:,2))>=i,1);  % 详解: 赋值：将 find(...) 的结果保存到 parent
                
                summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                for m = t:length(seq)  % 详解: for 循环：迭代变量 m 遍历 t:length(seq)
                    pro = logProd(beta(t,m,d,i), eta_out(m,d-1,parent));  % 详解: 赋值：将 logProd(...) 的结果保存到 pro
                    summ = logSum(summ,pro);  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                end  % 详解: 执行语句
                
                temp = logProd(summ, eta_in(t,d-1,parent));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                temp = logProd(temp, pi(parent,i,d-1));  % 详解: 赋值：将 logProd(...) 的结果保存到 temp
                chi(t,d,i) = logProd(temp, -POlambda);  % 详解: 调用函数：chi(t,d,i) = logProd(temp, -POlambda)
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句



for d = 2:max(allY)  % 详解: for 循环：迭代变量 d 遍历 2:max(allY)
    for i = 1:max(allX)  % 详解: for 循环：迭代变量 i 遍历 1:max(allX)
        if q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,1)==1)
            jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
            for t = 2:length(seq)  % 详解: for 循环：迭代变量 t 遍历 2:length(seq)
                summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                for k = jArray(1:end-1)  % 详解: for 循环：迭代变量 k 遍历 jArray(1:end-1)
                    summ = logSum(summ, xi(t-1,d,k,d,i));  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                end  % 详解: 执行语句
                gamma_in(t,d,i) = summ;  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

for d = 2:max(allY)  % 详解: for 循环：迭代变量 d 遍历 2:max(allY)
    for i = 1:max(allX)  % 详解: for 循环：迭代变量 i 遍历 1:max(allX)
        if q(d,i,1)==1  % 详解: 条件判断：if (q(d,i,1)==1)
            jArray = findJArray(q,d,i);  % 详解: 赋值：将 findJArray(...) 的结果保存到 jArray
            for t = 1:(length(seq))  % 详解: for 循环：迭代变量 t 遍历 1:(length(seq))
                summ = NaN;  % 详解: 赋值：计算表达式并保存到 summ
                for k = jArray  % 详解: for 循环：迭代变量 k 遍历 jArray
                    summ = logSum(summ, xi(t,d,i,d,k));  % 详解: 赋值：将 logSum(...) 的结果保存到 summ
                end  % 详解: 执行语句
                gamma_out(t,d,i)= summ;  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句


end  % 详解: 执行语句
    



