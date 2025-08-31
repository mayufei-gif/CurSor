% 文件: vqsplit.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [m, p, DistHist,minIndx]=vqsplit(X,L)  % 详解: 函数定义：vqsplit(X,L), 返回：m, p, DistHist,minIndx

e=.01;  % 详解: 赋值：计算表达式并保存到 e
eRed=0.75;  % 详解: 赋值：计算表达式并保存到 eRed
DT=.005;  % 详解: 赋值：计算表达式并保存到 DT
DTRed=0.75;  % 详解: 赋值：计算表达式并保存到 DTRed
MinPop=0.10;  % 详解: 赋值：计算表达式并保存到 MinPop


d=size(X,1);  % 详解: 赋值：将 size(...) 的结果保存到 d
N=size(X,2);  % 详解: 赋值：将 size(...) 的结果保存到 N
isFirstRound=1;  % 详解: 赋值：计算表达式并保存到 isFirstRound

if numel(L)==1  % 详解: 条件判断：if (numel(L)==1)
    M=mean(X,2);  % 详解: 赋值：将 mean(...) 的结果保存到 M
    CB=[M*(1+e) M*(1-e)];  % 详解: 赋值：计算表达式并保存到 CB
else  % 详解: 条件判断：else 分支
    CB=L;  % 详解: 赋值：计算表达式并保存到 CB
    L=size(CB,2);  % 详解: 赋值：将 size(...) 的结果保存到 L
    e=e*(eRed^fix(log2(L)));  % 详解: 赋值：计算表达式并保存到 e
    DT=DT*(DTRed^fix(log2(L)));  % 详解: 赋值：计算表达式并保存到 DT
end  % 详解: 执行语句

LC=size(CB,2);  % 详解: 赋值：将 size(...) 的结果保存到 LC

Iter=0;  % 详解: 赋值：计算表达式并保存到 Iter
Split=0;  % 详解: 赋值：计算表达式并保存到 Split
IsThereABestCB=0;  % 详解: 赋值：计算表达式并保存到 IsThereABestCB
maxIterInEachSize=20;  % 详解: 赋值：计算表达式并保存到 maxIterInEachSize
EachSizeIterCounter=0;  % 详解: 赋值：计算表达式并保存到 EachSizeIterCounter
while 1  % 详解: while 循环：当 (1) 为真时迭代
    [minIndx, dst]=VQIndex(X,CB);  % 详解: 执行语句

    ClusterD=zeros(1,LC);  % 详解: 赋值：将 zeros(...) 的结果保存到 ClusterD
    Population=zeros(1,LC);  % 详解: 赋值：将 zeros(...) 的结果保存到 Population
    LowPop=[];  % 详解: 赋值：计算表达式并保存到 LowPop
    for i=1:LC  % 详解: for 循环：迭代变量 i 遍历 1:LC
        Ind=find(minIndx==i);  % 详解: 赋值：将 find(...) 的结果保存到 Ind
        if length(Ind)<MinPop*N/LC  % 详解: 条件判断：if (length(Ind)<MinPop*N/LC)
            LowPop=[LowPop i];  % 详解: 赋值：计算表达式并保存到 LowPop
        else  % 详解: 条件判断：else 分支
            CB(:,i)=mean(X(:,Ind),2);  % 详解: 调用函数：CB(:,i)=mean(X(:,Ind),2)
            Population(i)=length(Ind);  % 详解: 调用函数：Population(i)=length(Ind)
            ClusterD(i)=sum(dst(Ind));  % 详解: 调用函数：ClusterD(i)=sum(dst(Ind))
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    if ~isempty(LowPop)  % 详解: 条件判断：if (~isempty(LowPop))
        [temp MaxInd]=maxn(Population,length(LowPop));  % 详解: 获取向量/矩阵尺寸
        CB(:,LowPop)=CB(:,MaxInd)*(1+e);  % 详解: 调用函数：CB(:,LowPop)=CB(:,MaxInd)*(1+e)
        CB(:,MaxInd)=CB(:,MaxInd)*(1-e);  % 详解: 调用函数：CB(:,MaxInd)=CB(:,MaxInd)*(1-e)
        
        [minIndx, dst]=VQIndex(X,CB);  % 详解: 执行语句

        ClusterD=zeros(1,LC);  % 详解: 赋值：将 zeros(...) 的结果保存到 ClusterD
        Population=zeros(1,LC);  % 详解: 赋值：将 zeros(...) 的结果保存到 Population
        
        for i=1:LC  % 详解: for 循环：迭代变量 i 遍历 1:LC
            Ind=find(minIndx==i);  % 详解: 赋值：将 find(...) 的结果保存到 Ind
            if ~isempty(Ind)  % 详解: 条件判断：if (~isempty(Ind))
                CB(:,i)=mean(X(:,Ind),2);  % 详解: 调用函数：CB(:,i)=mean(X(:,Ind),2)
                Population(i)=length(Ind);  % 详解: 调用函数：Population(i)=length(Ind)
                ClusterD(i)=sum(dst(Ind));  % 详解: 调用函数：ClusterD(i)=sum(dst(Ind))
            else  % 详解: 条件判断：else 分支
                CB(:,i)=X(:,fix(rand*N)+1);  % 详解: 调用函数：CB(:,i)=X(:,fix(rand*N)+1)
                disp('A random vector was assigned as a codeword.')  % 详解: 调用函数：disp('A random vector was assigned as a codeword.')
                isFirstRound=1;  % 详解: 赋值：计算表达式并保存到 isFirstRound
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    Iter=Iter+1;  % 详解: 赋值：计算表达式并保存到 Iter
    if isFirstRound  % 详解: 条件判断：if (isFirstRound)
        TotalDist=sum(ClusterD(~isnan(ClusterD)));  % 详解: 赋值：将 sum(...) 的结果保存到 TotalDist
        DistHist(Iter)=TotalDist;  % 详解: 执行语句
        PrevTotalDist=TotalDist;  % 详解: 赋值：计算表达式并保存到 PrevTotalDist
        isFirstRound=0;  % 详解: 赋值：计算表达式并保存到 isFirstRound
    else  % 详解: 条件判断：else 分支
        TotalDist=sum(ClusterD(~isnan(ClusterD)));  % 详解: 赋值：将 sum(...) 的结果保存到 TotalDist
        DistHist(Iter)=TotalDist;  % 详解: 执行语句
        PercentageImprovement=((PrevTotalDist-TotalDist)/PrevTotalDist);  % 详解: 赋值：计算表达式并保存到 PercentageImprovement
        if PercentageImprovement>=DT  % 详解: 条件判断：if (PercentageImprovement>=DT)
            PrevTotalDist=TotalDist;  % 详解: 赋值：计算表达式并保存到 PrevTotalDist
            isFirstRound=0;  % 详解: 赋值：计算表达式并保存到 isFirstRound
        else  % 详解: 条件判断：else 分支
            EachSizeIterCounter=0;  % 详解: 赋值：计算表达式并保存到 EachSizeIterCounter
            if LC>=L  % 详解: 条件判断：if (LC>=L)
                if L==LC  % 详解: 条件判断：if (L==LC)
                    disp(TotalDist)  % 详解: 调用函数：disp(TotalDist)
                    break  % 详解: 跳出循环：break
                else  % 详解: 条件判断：else 分支
                    [temp, Ind]=min(Population);  % 详解: 统计：最大/最小值
                    NCB=zeros(d,LC-1);  % 详解: 赋值：将 zeros(...) 的结果保存到 NCB
                    NCB=CB(:,setxor(1:LC,Ind(1)));  % 详解: 赋值：将 CB(...) 的结果保存到 NCB
                    CB=NCB;  % 详解: 赋值：计算表达式并保存到 CB
                    LC=LC-1;  % 详解: 赋值：计算表达式并保存到 LC
                    isFirstRound=1;  % 详解: 赋值：计算表达式并保存到 isFirstRound
                end  % 详解: 执行语句
            else  % 详解: 条件判断：else 分支
                CB=[CB*(1+e) CB*(1-e)];  % 详解: 赋值：计算表达式并保存到 CB
                e=eRed*e;  % 详解: 赋值：计算表达式并保存到 e
                DT=DT*DTRed;  % 详解: 赋值：计算表达式并保存到 DT
                LC=size(CB,2);  % 详解: 赋值：将 size(...) 的结果保存到 LC
                isFirstRound=1;  % 详解: 赋值：计算表达式并保存到 isFirstRound
                Split=Split+1;  % 详解: 赋值：计算表达式并保存到 Split
                IsThereABestCB=0;  % 详解: 赋值：计算表达式并保存到 IsThereABestCB
                disp(LC)  % 详解: 调用函数：disp(LC)
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    if ~IsThereABestCB  % 详解: 条件判断：if (~IsThereABestCB)
        BestCB=CB;  % 详解: 赋值：计算表达式并保存到 BestCB
        BestD=TotalDist;  % 详解: 赋值：计算表达式并保存到 BestD
        IsThereABestCB=1;  % 详解: 赋值：计算表达式并保存到 IsThereABestCB
    else  % 详解: 条件判断：else 分支
        if TotalDist<BestD  % 详解: 条件判断：if (TotalDist<BestD)
            BestCB=CB;  % 详解: 赋值：计算表达式并保存到 BestCB
            BestD=TotalDist;  % 详解: 赋值：计算表达式并保存到 BestD
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    EachSizeIterCounter=EachSizeIterCounter+1;  % 详解: 赋值：计算表达式并保存到 EachSizeIterCounter
    if EachSizeIterCounter>maxIterInEachSize  % 详解: 条件判断：if (EachSizeIterCounter>maxIterInEachSize)
        EachSizeIterCounter=0;  % 详解: 赋值：计算表达式并保存到 EachSizeIterCounter
        CB=BestCB;  % 详解: 赋值：计算表达式并保存到 CB
        IsThereABestCB=0;  % 详解: 赋值：计算表达式并保存到 IsThereABestCB
        if LC>=L  % 详解: 条件判断：if (LC>=L)
            if L==LC  % 详解: 条件判断：if (L==LC)
                disp(TotalDist)  % 详解: 调用函数：disp(TotalDist)
                break  % 详解: 跳出循环：break
            else  % 详解: 条件判断：else 分支
                [temp, Ind]=min(Population);  % 详解: 统计：最大/最小值
                NCB=zeros(d,LC-1);  % 详解: 赋值：将 zeros(...) 的结果保存到 NCB
                NCB=CB(:,setxor(1:LC,Ind(1)));  % 详解: 赋值：将 CB(...) 的结果保存到 NCB
                CB=NCB;  % 详解: 赋值：计算表达式并保存到 CB
                LC=LC-1;  % 详解: 赋值：计算表达式并保存到 LC
                isFirstRound=1;  % 详解: 赋值：计算表达式并保存到 isFirstRound
            end  % 详解: 执行语句
        else  % 详解: 条件判断：else 分支
            CB=[CB*(1+e) CB*(1-e)];  % 详解: 赋值：计算表达式并保存到 CB
            e=eRed*e;  % 详解: 赋值：计算表达式并保存到 e
            DT=DT*DTRed;  % 详解: 赋值：计算表达式并保存到 DT
            LC=size(CB,2);  % 详解: 赋值：将 size(...) 的结果保存到 LC
            isFirstRound=1;  % 详解: 赋值：计算表达式并保存到 isFirstRound
            Split=Split+1;  % 详解: 赋值：计算表达式并保存到 Split
            IsThereABestCB=0;  % 详解: 赋值：计算表达式并保存到 IsThereABestCB
            disp(LC)  % 详解: 调用函数：disp(LC)
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    disp(TotalDist)  % 详解: 调用函数：disp(TotalDist)
    p=Population/N;  % 详解: 赋值：计算表达式并保存到 p
    save CBTemp CB p DistHist  % 详解: 执行语句
end  % 详解: 执行语句
m=CB;  % 详解: 赋值：计算表达式并保存到 m

p=Population/N;  % 详解: 赋值：计算表达式并保存到 p

disp(['Iterations = ' num2str(Iter)]);  % 详解: 调用函数：disp(['Iterations = ' num2str(Iter)])
disp(['Split = ' num2str(Split)]);  % 详解: 调用函数：disp(['Split = ' num2str(Split)])

function [v, i]=maxn(x,n)  % 详解: 函数定义：maxn(x,n), 返回：v, i

if nargin<2  % 详解: 条件判断：if (nargin<2)
    [v, i]=max(x);  % 详解: 统计：最大/最小值
else  % 详解: 条件判断：else 分支
    n=min(length(x),n);  % 详解: 赋值：将 min(...) 的结果保存到 n
    [v, i]=sort(x);  % 详解: 执行语句
    v=v(end:-1:end-n+1);  % 详解: 赋值：将 v(...) 的结果保存到 v
    i=i(end:-1:end-n+1);  % 详解: 赋值：将 i(...) 的结果保存到 i
end  % 详解: 执行语句
        
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
    
function [I, dist]=VQLSFSpectralIndex(X,CB,W)  % 详解: 函数定义：VQLSFSpectralIndex(X,CB,W), 返回：I, dist

if nargin<3  % 详解: 条件判断：if (nargin<3)
    L=256;  % 详解: 赋值：计算表达式并保存到 L
    W=ones(L,1);  % 详解: 赋值：将 ones(...) 的结果保存到 W
else  % 详解: 条件判断：else 分支
    if isscalar(W)  % 详解: 条件判断：if (isscalar(W))
        L=W;  % 详解: 赋值：计算表达式并保存到 L
        W=ones(L,1);  % 详解: 赋值：将 ones(...) 的结果保存到 W
    elseif isvector(W)  % 详解: 条件判断：elseif (isvector(W))
        W=W(:);  % 详解: 赋值：将 W(...) 的结果保存到 W
        L=length(W);  % 详解: 赋值：将 length(...) 的结果保存到 L
    else  % 详解: 条件判断：else 分支
        error('Invalid input argument. W should be either a vector or a scaler!')  % 详解: 调用函数：error('Invalid input argument. W should be either a vector or a scaler!')
    end  % 详解: 执行语句
end  % 详解: 执行语句

NX=size(X,2);  % 详解: 赋值：将 size(...) 的结果保存到 NX
NCB=size(CB,2);  % 详解: 赋值：将 size(...) 的结果保存到 NCB

AX=lsf2lpc(X);  % 详解: 赋值：将 lsf2lpc(...) 的结果保存到 AX
ACB=lsf2lpc(CB);  % 详解: 赋值：将 lsf2lpc(...) 的结果保存到 ACB


D=zeros(NCB,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 D

w=linspace(0,pi,L+1);  % 详解: 赋值：将 linspace(...) 的结果保存到 w
w=w(1:end-1);  % 详解: 赋值：将 w(...) 的结果保存到 w
N=size(AX,2)-1;  % 详解: 赋值：将 size(...) 的结果保存到 N
WFZ=zeros(N+1,L);  % 详解: 赋值：将 zeros(...) 的结果保存到 WFZ
IMAGUNIT=sqrt(-1);  % 详解: 赋值：将 sqrt(...) 的结果保存到 IMAGUNIT
for k=0:N  % 详解: for 循环：迭代变量 k 遍历 0:N
    WFZ(k+1,:)=exp(IMAGUNIT*k*w);  % 详解: 调用函数：WFZ(k+1,:)=exp(IMAGUNIT*k*w)
end  % 详解: 执行语句

SCB=zeros(L,NCB);  % 详解: 赋值：将 zeros(...) 的结果保存到 SCB
for i=1:NCB  % 详解: for 循环：迭代变量 i 遍历 1:NCB
    SCB(:,i)=(1./abs(ACB(i,:)*WFZ));  % 详解: 调用函数：SCB(:,i)=(1./abs(ACB(i,:)*WFZ))
end  % 详解: 执行语句

I=zeros(1,NX);  % 详解: 赋值：将 zeros(...) 的结果保存到 I
dist=zeros(1,NX);  % 详解: 赋值：将 zeros(...) 的结果保存到 dist
for j=1:NX  % 详解: for 循环：迭代变量 j 遍历 1:NX
    SX=(1./abs(AX(j,:)*WFZ))';      % 赋值：设置变量 SX  % 详解: 赋值：计算表达式并保存到 SX  % 详解: 赋值：计算表达式并保存到 SX
    for i=1:NCB  % 详解: for 循环：迭代变量 i 遍历 1:NCB
        D(i)=sqrt(sum(((SX-SCB(:,i)).^2).*W));  % 详解: 调用函数：D(i)=sqrt(sum(((SX-SCB(:,i)).^2).*W))
    end  % 详解: 执行语句
    [dist(j), I(j)]=min(D);  % 详解: 统计：最大/最小值
end  % 详解: 执行语句



