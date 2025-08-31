% 文件: dhmm_em_demo.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

clear all  % 详解: 执行语句
load f5-1.txt  % 详解: 执行语句
for i=1:6  % 详解: for 循环：迭代变量 i 遍历 1:6
z(i,:)=f5_1((i-1)*512+1:512*i);  % 详解: 调用函数：z(i,:)=f5_1((i-1)*512+1:512*i)
end  % 详解: 执行语句
load f11-1.txt  % 详解: 执行语句
for i=1:6  % 详解: for 循环：迭代变量 i 遍历 1:6
z(6+i,:)=f11_1((i-1)*512+1:512*i);  % 详解: 调用函数：z(6+i,:)=f11_1((i-1)*512+1:512*i)
end  % 详解: 执行语句

O =10;  % 详解: 赋值：计算表达式并保存到 O
Q = 4;  % 详解: 赋值：计算表达式并保存到 Q

 N=64;  % 详解: 赋值：计算表达式并保存到 N
 deltaN=16;  % 详解: 赋值：计算表达式并保存到 deltaN
 M=12;  % 详解: 赋值：计算表达式并保存到 M
 Q1=12;  % 详解: 赋值：计算表达式并保存到 Q1
K=O;L=3;  % 详解: 赋值：计算表达式并保存到 K
maxiter=500;  % 详解: 赋值：计算表达式并保存到 maxiter
for j=1:L  % 详解: for 循环：迭代变量 j 遍历 1:L
 for i=1:4  % 详解: for 循环：迭代变量 i 遍历 1:4
   x=z(4*(j-1)+i,:)';  % 赋值：设置变量 x  % 详解: 赋值：将 z(...) 的结果保存到 x  % 详解: 赋值：将 z(...) 的结果保存到 x
    y=hmmfeatures(x,N,deltaN,M,Q1);  % 详解: 赋值：将 hmmfeatures(...) 的结果保存到 y
    [yc,c,errlog]= kmeans1(y,K,maxiter);  % 详解: 执行语句
    data(i,:,j)=c;  % 详解: 执行语句
 end  % 详解: 执行语句
prior1 = normalise(rand(Q,1));  % 详解: 赋值：将 normalise(...) 的结果保存到 prior1
transmat1 = mk_stochastic(rand(Q,Q));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 transmat1
obsmat1 = mk_stochastic(rand(Q,O));  % 详解: 赋值：将 mk_stochastic(...) 的结果保存到 obsmat1

[LL, prior2(:,:,j), transmat2(:,:,j), obsmat2(:,:,j)] = dhmm_em(data(:,:,j), prior1, transmat1, obsmat1, 'max_iter', 100);  % 详解: 执行语句
figure(1);  % 详解: 调用函数：figure(1)
subplot(3,1,j);  % 详解: 调用函数：subplot(3,1,j)
plot(LL);  % 详解: 调用函数：plot(LL)
end  % 详解: 执行语句
   
C1=data(1,:,2);  % 详解: 赋值：将 data(...) 的结果保存到 C1
for i=1:L  % 详解: for 循环：迭代变量 i 遍历 1:L
loglik = dhmm_logprob(C1, prior2(:,:,i), transmat2(:,:,i), obsmat2(:,:,i));  % 详解: 赋值：将 dhmm_logprob(...) 的结果保存到 loglik
b(i)=(loglik);  % 详解: 调用函数：b(i)=(loglik)
end  % 详解: 执行语句
disp('b=');  % 详解: 调用函数：disp('b=')
disp(b);  % 详解: 调用函数：disp(b)



