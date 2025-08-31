%BP������Ĺ�·������Ԥ�� P121  


%δdebug��
clc  %����
clear all;  %����ڴ��Լӿ������ٶ�
close all;   %�رյ�ǰ����figureͼ��
SamNum=20; %��������������Ϊ20
TestSamNum=20;%������������Ϊ20
ForcastSamNum=2; %Ԥ����������Ϊ2
HiddenUnitNum=8;%�м�����ڵ�����ȡ8
InDim=3;%��������ά��Ϊ��
OutDim=2;%�������Ϊά��Ϊ2
%ԭʼ����
%���������ˣ�
sqrs=[20.55 22.44 25.37 27.13 29.45 30.10 30.96 34.06 36.42 38.09 39.13 39.99 41.43 44.59 47.30 52.89 55.73 56.76 79.17 60.63];
%����������������
sqjdcs=[0.6 0.75 0.85 0.9 1.05 1.35 1.45 1.6 1.7 1.85 2.15 2.2 2.25 2.35 2.5 2.6 2.7 2.85 2.95 3.1];
%��·�������ƽ��ǧ�ף�
sqglmj=[0.09 0.11 0.11 0.14 0.20 0.23 0.23 0.32 0.32 0.34 0.36 0.36 0.38 0.49 0.56 0.59 0.59 0.67 0.69 0.79];
%��·�����������ˣ�
glkyl=[5126 6217 7730 9145 10460 11387 12353 15750 18304 19836 21024 19490 20433 22598 25107 33442 36836 40548 42927 43462];
%��·����������֣�
glhyl=[1237 1379 1385 1399 1663 1714 1834 4322 8132 8936 11099 11203 10524 11115 13320 16762 18673 20724 20803 21804];
p=[sqrs;sqjdcs;sqglmj]; %�������ݾ���
t=[glkyl;glhyl];         %Ŀ�����ݾ���
[SamIn,minp,maxp,tn,mint,maxt]=premnmx(p,t); %ԭʼ�����ԣ��������������ʼ��


rand('state',sum(100*clock));        %����ϵͳʱ�����Ӳ��������
NoiseVar=0.01;                       %����ǿ��Ϊ0.01�����������ԭ���Ƿ�ֹ���������ϣ�
Noise=NoiseVar*randn(2,SamNum);     %��������
SamOut=tn+Noise;                  %�������ӵ����������

TestSamIn=SamIn;

TestOut=SamOut;

MaxEpochs=50000;
lr=0.035;
E0=0.65*10^(-3);
W1=0.5*rand(HiddenUnitNum,InDim)-0.1;
B1=0.5*rand(HiddenUnitNum,1)-0.1;
W2=0.5*rand(OutDim,HiddenUnitNum)-0.1;
B2=0.5*rand(OutDim,1)-0.1;
ErrHistory=[];

for i=1:MaxEpochs
        HiddenOut=logsig(W1*SamIn+repmat(B1,1,SamNum));
        NetWorkOut=W2*HiddenOut+repmat(B2,1,SamNum);
        Error=SamOut-NetWorkOut;
        SSE=sumsqr(Error)
       
        ErrHistory=[ErrHistory SSE];
        if SSE<E0,break,end
        
        Delta2=Error;
        Deltal= W2'* Delta2.*HiddenOut.*(1-HiddenOut);
        
        dW2=Delta2*HiddenOut';
        dB2=Delta2*ones(SamNum,1);
        
        
        dW1=Deltal*SamIn';
        dB1=Deltal*ones(SamNum,1);
        
        W2=W2+lr*dW2;
        B2=B2+lr*dB2;
        
        W1=W1+lr*dW1;
        B1=B1+lr*dB1;
end


HiddenOut=logsig(W1*SamIn+repmat(B1,1,TestSamNum));
NetworkOut=W2*HiddenOut+repmat(B2,1,TestSamNum);
a=postmnmx(NetworkOut,mint,maxt);
x=1990:2009;
newk=a(1,:);
newh=a(2,:);
figure;
subplot(2,1,1);plot(x,newk,'r-o',x,glkyl,'b--+');
legend('�������������','ʵ�ʿ�����');
xlabel('���');ylabel('������/����');
title('Դ���������������ѧϰ�Ͳ��ԶԱ�ͼ');
title('Դ���������������ѧϰ�Ͳ��ԶԱ�ͼ');
subplot(2,1,2);plot(x,newh,'r-o',x,glhyl,'b--+');
legend('�������������','ʵ�ʻ�����');
xlabel('���');ylabel('������/���');

%����ѵ���õ�������в���
pnew=[73,39  75.55  
     3.9635  4.097 
     0.9880  1.0268];
pnewm=tramnmx(pnew,minp,maxp);


HiddenOut=logsig(W1*pnew+repmat(B1,1,ForcastSamNum));
anewn=W2*HiddenOut+repmat(B1,1,ForcastSamNum);


anew=postmnmx(anewn,mint,maxt);
