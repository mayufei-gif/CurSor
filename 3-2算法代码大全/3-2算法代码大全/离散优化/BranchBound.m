% 文件: BranchBound.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%%本程序是用分枝定界法求解整数线性规划问题
%%问题的标准形式：
%%  min c'*x  % 中文: ％ 英石。 a*x <= b
%% s.t. A*x<=b  % 中文: ％aeq*x = beq |||英石。  2x1+x2 <= 8
%%      Aeq*x=beq  % 中文: 2x1+3x2 = 9;
%%  x要求是整数
%例 min Z=x1+4x2
% s.t.  2x1+x2<=8  % 中文: C = [1 4];
%       x1+2x2>=6
%       2x1+3x2=9;  % 中文: a = [2 1; -1 -2];
%       x1, x2>=0且为整数
%先将x1+2x2>=6化为 - x1 - 2x2<= -6
%c=[1 4];  % 中文: b = [8; -6];
%A=[2 1;-1 -2];  % 中文: aeq = [2 3];
%b=[8;-6];  % 中文: beq = 9;
%Aeq=[2 3];  % 中文: [y，fval] = branchbound（c，a，b，aeq，beq）|||清除; map = [1 2 30; 2 4 5; 2 5 50; 3 2 6; 4 3 1; 1 4 20; 1 5 3]
%beq=9;  % 中文: [P，V] = Dijkstra（Map，2,5）|||另请参见Kruskal，LPINT，DP，BNBGUI，BNB18，||| W. Z. Li，2000
%[y,fval]=BranchBound(c,A,b,Aeq,beq)  % 中文: [p_opt，fval] = dynprog（x，decisfun，objfun，transfun）||| eg13f1_2.m |||函数u = decISF_1（k，x）||| c = [70,72,80,76]; q = 10*[6,7,12,6];



function [y,fval]=BranchBound(c,A,b,Aeq,beq)  % 详解: 函数定义：BranchBound(c,A,b,Aeq,beq), 返回：y,fval
NL=length(c);  % 详解: 赋值：将 length(...) 的结果保存到 NL
UB=inf;  % 详解: 赋值：计算表达式并保存到 UB
LB=-inf;  % 详解: 赋值：计算表达式并保存到 LB
FN=[0];  % 详解: 赋值：计算表达式并保存到 FN
AA(1)={A};  % 详解: 执行语句
BB(1)={b};  % 详解: 执行语句
k=0;  % 详解: 赋值：计算表达式并保存到 k
flag=0;  % 详解: 赋值：计算表达式并保存到 flag
while flag==0;  % 详解: while 循环：当 (flag==0;) 为真时迭代
    [x,fval,exitFlag]=linprog(c,A,b,Aeq,beq);  % 详解: 执行语句
    if (exitFlag == -2) | (fval >= UB)  % 详解: 条件判断：if ((exitFlag == -2) | (fval >= UB))
        FN(1)=[];  % 详解: 执行语句
        if isempty(FN)==1  % 详解: 条件判断：if (isempty(FN)==1)
            flag=1;  % 详解: 赋值：计算表达式并保存到 flag
        else  % 详解: 条件判断：else 分支
            k=FN(1);  % 详解: 赋值：将 FN(...) 的结果保存到 k
            A=AA{k};  % 详解: 赋值：计算表达式并保存到 A
            b=BB{k};  % 详解: 赋值：计算表达式并保存到 b
        end  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
        for i=1:NL  % 详解: for 循环：迭代变量 i 遍历 1:NL
            if abs(x(i)-round(x(i)))>1e-7  % 详解: 条件判断：if (abs(x(i)-round(x(i)))>1e-7)
                kk=FN(end);  % 详解: 赋值：将 FN(...) 的结果保存到 kk
                FN=[FN,kk+1,kk+2];  % 详解: 赋值：计算表达式并保存到 FN
                temp_A=zeros(1,NL);  % 详解: 赋值：将 zeros(...) 的结果保存到 temp_A
                temp_A(i)=1;  % 详解: 执行语句
                temp_A1=[A;temp_A];  % 详解: 赋值：计算表达式并保存到 temp_A1
                AA(kk+1)={temp_A1};  % 详解: 执行语句
                b1=[b;fix(x(i))];  % 详解: 赋值：计算表达式并保存到 b1
                BB(kk+1)={b1};  % 详解: 执行语句
                temp_A2=[A;-temp_A];  % 详解: 赋值：计算表达式并保存到 temp_A2
                AA(kk+2)={temp_A2};  % 详解: 执行语句
                b2=[b;-(fix(x(i))+1)];  % 详解: 赋值：计算表达式并保存到 b2
                BB(kk+2)={b2};  % 详解: 执行语句
                FN(1)=[];  % 详解: 执行语句
                k=FN(1);  % 详解: 赋值：将 FN(...) 的结果保存到 k
                A=AA{k};  % 详解: 赋值：计算表达式并保存到 A
                b=BB{k};  % 详解: 赋值：计算表达式并保存到 b
                break;  % 详解: 跳出循环：break
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        if (i==NL) & (abs(x(i)-round(x(i)))<=1e-7)  % 详解: 条件判断：if ((i==NL) & (abs(x(i)-round(x(i)))<=1e-7))
            UB=fval;  % 详解: 赋值：计算表达式并保存到 UB
            y=x;  % 详解: 赋值：计算表达式并保存到 y
            FN(1)=[];  % 详解: 执行语句
            if isempty(FN)==1  % 详解: 条件判断：if (isempty(FN)==1)
                flag=1;  % 详解: 赋值：计算表达式并保存到 flag
            else  % 详解: 条件判断：else 分支
                k=FN(1);  % 详解: 赋值：将 FN(...) 的结果保存到 k
                A=AA{k};  % 详解: 赋值：计算表达式并保存到 A
                b=BB{k};  % 详解: 赋值：计算表达式并保存到 b
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句
y=round(y);  % 详解: 赋值：将 round(...) 的结果保存到 y
fval=c*y;  % 详解: 赋值：计算表达式并保存到 fval




