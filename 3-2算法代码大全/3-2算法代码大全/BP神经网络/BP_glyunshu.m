% 文件: BP_glyunshu.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%BP神经网络的公路运量的预测 P121  


%未debug完
clc  % 详解: 执行语句
clear all;  % 详解: 执行语句
close all;  % 详解: 执行语句
SamNum=20;  % 详解: 赋值：计算表达式并保存到 SamNum
TestSamNum=20;  % 详解: 赋值：计算表达式并保存到 TestSamNum
ForcastSamNum=2;  % 详解: 赋值：计算表达式并保存到 ForcastSamNum
HiddenUnitNum=8;  % 详解: 赋值：计算表达式并保存到 HiddenUnitNum
InDim=3;  % 详解: 赋值：计算表达式并保存到 InDim
OutDim=2;  % 详解: 赋值：计算表达式并保存到 OutDim
sqrs=[20.55 22.44 25.37 27.13 29.45 30.10 30.96 34.06 36.42 38.09 39.13 39.99 41.43 44.59 47.30 52.89 55.73 56.76 79.17 60.63];  % 详解: 赋值：计算表达式并保存到 sqrs
sqjdcs=[0.6 0.75 0.85 0.9 1.05 1.35 1.45 1.6 1.7 1.85 2.15 2.2 2.25 2.35 2.5 2.6 2.7 2.85 2.95 3.1];  % 详解: 赋值：计算表达式并保存到 sqjdcs
sqglmj=[0.09 0.11 0.11 0.14 0.20 0.23 0.23 0.32 0.32 0.34 0.36 0.36 0.38 0.49 0.56 0.59 0.59 0.67 0.69 0.79];  % 详解: 赋值：计算表达式并保存到 sqglmj
glkyl=[5126 6217 7730 9145 10460 11387 12353 15750 18304 19836 21024 19490 20433 22598 25107 33442 36836 40548 42927 43462];  % 详解: 赋值：计算表达式并保存到 glkyl
glhyl=[1237 1379 1385 1399 1663 1714 1834 4322 8132 8936 11099 11203 10524 11115 13320 16762 18673 20724 20803 21804];  % 详解: 赋值：计算表达式并保存到 glhyl
p=[sqrs;sqjdcs;sqglmj];  % 详解: 赋值：计算表达式并保存到 p
t=[glkyl;glhyl];  % 详解: 赋值：计算表达式并保存到 t
[SamIn,minp,maxp,tn,mint,maxt]=premnmx(p,t);  % 详解: 执行语句


rand('state',sum(100*clock));  % 详解: 调用函数：rand('state',sum(100*clock))
NoiseVar=0.01;  % 详解: 赋值：计算表达式并保存到 NoiseVar
Noise=NoiseVar*randn(2,SamNum);  % 详解: 赋值：计算表达式并保存到 Noise
SamOut=tn+Noise;  % 详解: 赋值：计算表达式并保存到 SamOut

TestSamIn=SamIn;  % 详解: 赋值：计算表达式并保存到 TestSamIn

TestOut=SamOut;  % 详解: 赋值：计算表达式并保存到 TestOut

MaxEpochs=50000;  % 详解: 赋值：计算表达式并保存到 MaxEpochs
lr=0.035;  % 详解: 赋值：计算表达式并保存到 lr
E0=0.65*10^(-3);  % 详解: 赋值：计算表达式并保存到 E0
W1=0.5*rand(HiddenUnitNum,InDim)-0.1;  % 详解: 赋值：计算表达式并保存到 W1
B1=0.5*rand(HiddenUnitNum,1)-0.1;  % 详解: 赋值：计算表达式并保存到 B1
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1;  % 详解: 赋值：计算表达式并保存到 W2
B2=0.5*rand(OutDim,1)-0.1;  % 详解: 赋值：计算表达式并保存到 B2
ErrHistory=[];  % 详解: 赋值：计算表达式并保存到 ErrHistory

for i=1:MaxEpochs  % 详解: for 循环：迭代变量 i 遍历 1:MaxEpochs
        HiddenOut=logsig(W1*SamIn+repmat(B1,1,SamNum));  % 详解: 赋值：将 logsig(...) 的结果保存到 HiddenOut
        NetWorkOut=W2*HiddenOut+repmat(B2,1,SamNum);  % 详解: 赋值：计算表达式并保存到 NetWorkOut
        Error=SamOut-NetWorkOut;  % 详解: 赋值：计算表达式并保存到 Error
        SSE=sumsqr(Error)  % 详解: 赋值：将 sumsqr(...) 的结果保存到 SSE
       
        ErrHistory=[ErrHistory SSE];  % 详解: 赋值：计算表达式并保存到 ErrHistory
        if SSE<E0,break,end  % 详解: 条件判断：if (SSE<E0,break,end)
        
        Delta2=Error;  % 详解: 赋值：计算表达式并保存到 Delta2
        Deltal= W2'* Delta2.*HiddenOut.*(1-HiddenOut);  % 赋值：设置变量 Deltal  % 详解: 赋值：计算表达式并保存到 Deltal  % 详解: 赋值：计算表达式并保存到 Deltal
        
        dW2=Delta2*HiddenOut';  % 赋值：设置变量 dW2  % 详解: 赋值：计算表达式并保存到 dW2  % 详解: 赋值：计算表达式并保存到 dW2
        dB2=Delta2*ones(SamNum,1);  % 详解: 赋值：计算表达式并保存到 dB2
        
        
        dW1=Deltal*SamIn';  % 赋值：设置变量 dW1  % 详解: 赋值：计算表达式并保存到 dW1  % 详解: 赋值：计算表达式并保存到 dW1
        dB1=Deltal*ones(SamNum,1);  % 详解: 赋值：计算表达式并保存到 dB1
        
        W2=W2+lr*dW2;  % 详解: 赋值：计算表达式并保存到 W2
        B2=B2+lr*dB2;  % 详解: 赋值：计算表达式并保存到 B2
        
        W1=W1+lr*dW1;  % 详解: 赋值：计算表达式并保存到 W1
        B1=B1+lr*dB1;  % 详解: 赋值：计算表达式并保存到 B1
end  % 详解: 执行语句


HiddenOut=logsig(W1*SamIn+repmat(B1,1,TestSamNum));  % 详解: 赋值：将 logsig(...) 的结果保存到 HiddenOut
NetworkOut=W2*HiddenOut+repmat(B2,1,TestSamNum);  % 详解: 赋值：计算表达式并保存到 NetworkOut
a=postmnmx(NetworkOut,mint,maxt);  % 详解: 赋值：将 postmnmx(...) 的结果保存到 a
x=1990:2009;  % 详解: 赋值：计算表达式并保存到 x
newk=a(1,:);  % 详解: 赋值：将 a(...) 的结果保存到 newk
newh=a(2,:);  % 详解: 赋值：将 a(...) 的结果保存到 newh
figure;  % 详解: 执行语句
subplot(2,1,1);plot(x,newk,'r-o',x,glkyl,'b--+');  % 详解: 调用函数：subplot(2,1,1);plot(x,newk,'r-o',x,glkyl,'b--+')
legend('网络输出客运量','实际客运量');  % 详解: 调用函数：legend('网络输出客运量','实际客运量')
xlabel('年份');ylabel('客运量/万人');  % 详解: 调用函数：xlabel('年份');ylabel('客运量/万人')
title('源程序神经网络客运量学习和测试对比图');  % 详解: 调用函数：title('源程序神经网络客运量学习和测试对比图')
title('源程序神经网络货运量学习和测试对比图');  % 详解: 调用函数：title('源程序神经网络货运量学习和测试对比图')
subplot(2,1,2);plot(x,newh,'r-o',x,glhyl,'b--+');  % 详解: 调用函数：subplot(2,1,2);plot(x,newh,'r-o',x,glhyl,'b--+')
legend('网络输出货运量','实际货运量');  % 详解: 调用函数：legend('网络输出货运量','实际货运量')
xlabel('年份');ylabel('货运量/万吨');  % 详解: 调用函数：xlabel('年份');ylabel('货运量/万吨')

pnew=[73,39  75.55  % 详解: 赋值：计算表达式并保存到 pnew
     3.9635  4.097  % 详解: 执行语句
     0.9880  1.0268];  % 详解: 执行语句
pnewm=tramnmx(pnew,minp,maxp);  % 详解: 赋值：将 tramnmx(...) 的结果保存到 pnewm


HiddenOut=logsig(W1*pnew+repmat(B1,1,ForcastSamNum));  % 详解: 赋值：将 logsig(...) 的结果保存到 HiddenOut
anewn=W2*HiddenOut+repmat(B1,1,ForcastSamNum);  % 详解: 赋值：计算表达式并保存到 anewn


anew=postmnmx(anewn,mint,maxt);  % 详解: 赋值：将 postmnmx(...) 的结果保存到 anew




