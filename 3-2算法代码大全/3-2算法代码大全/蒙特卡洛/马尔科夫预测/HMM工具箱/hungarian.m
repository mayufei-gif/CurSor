% 文件: hungarian.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [C,T]=hungarian(A)  % 详解: 函数定义：hungarian(A), 返回：C,T




[m,n]=size(A);  % 详解: 获取向量/矩阵尺寸

if (m~=n)  % 详解: 条件判断：if ((m~=n))
    error('HUNGARIAN: Cost matrix must be square!');  % 详解: 调用函数：error('HUNGARIAN: Cost matrix must be square!')
end  % 详解: 执行语句

orig=A;  % 详解: 赋值：计算表达式并保存到 orig

A=hminired(A);  % 详解: 赋值：将 hminired(...) 的结果保存到 A

[A,C,U]=hminiass(A);  % 详解: 执行语句

while (U(n+1))  % 详解: while 循环：当 ((U(n+1))) 为真时迭代
    LR=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 LR
    LC=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 LC
    CH=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 CH
    RH=[zeros(1,n) -1];  % 详解: 赋值：计算表达式并保存到 RH
    
    SLC=[];  % 详解: 赋值：计算表达式并保存到 SLC
    
    r=U(n+1);  % 详解: 赋值：将 U(...) 的结果保存到 r
    LR(r)=-1;  % 详解: 执行语句
    SLR=r;  % 详解: 赋值：计算表达式并保存到 SLR
    
    while (1)  % 详解: while 循环：当 ((1)) 为真时迭代
        if (A(r,n+1)~=0)  % 详解: 条件判断：if ((A(r,n+1)~=0))
            l=-A(r,n+1);  % 详解: 赋值：计算表达式并保存到 l
            
            if (A(r,l)~=0 & RH(r)==0)  % 详解: 条件判断：if ((A(r,l)~=0 & RH(r)==0))
                RH(r)=RH(n+1);  % 详解: 调用函数：RH(r)=RH(n+1)
                RH(n+1)=r;  % 详解: 执行语句
                
                CH(r)=-A(r,l);  % 详解: 调用函数：CH(r)=-A(r,l)
            end  % 详解: 执行语句
        else  % 详解: 条件判断：else 分支
            if (RH(n+1)<=0)  % 详解: 条件判断：if ((RH(n+1)<=0))
                [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);  % 详解: 执行语句
            end  % 详解: 执行语句
            
            r=RH(n+1);  % 详解: 赋值：将 RH(...) 的结果保存到 r
            l=CH(r);  % 详解: 赋值：将 CH(...) 的结果保存到 l
            CH(r)=-A(r,l);  % 详解: 调用函数：CH(r)=-A(r,l)
            if (A(r,l)==0)  % 详解: 条件判断：if ((A(r,l)==0))
                RH(n+1)=RH(r);  % 详解: 调用函数：RH(n+1)=RH(r)
                RH(r)=0;  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        
        while (LC(l)~=0)  % 详解: while 循环：当 ((LC(l)~=0)) 为真时迭代
            if (RH(r)==0)  % 详解: 条件判断：if ((RH(r)==0))
                if (RH(n+1)<=0)  % 详解: 条件判断：if ((RH(n+1)<=0))
                    [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR);  % 详解: 执行语句
                end  % 详解: 执行语句
                
                r=RH(n+1);  % 详解: 赋值：将 RH(...) 的结果保存到 r
            end  % 详解: 执行语句
            
            l=CH(r);  % 详解: 赋值：将 CH(...) 的结果保存到 l
            
            CH(r)=-A(r,l);  % 详解: 调用函数：CH(r)=-A(r,l)
            
            if(A(r,l)==0)  % 详解: 调用函数：if(A(r,l)==0)
                RH(n+1)=RH(r);  % 详解: 调用函数：RH(n+1)=RH(r)
                RH(r)=0;  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        
        if (C(l)==0)  % 详解: 条件判断：if ((C(l)==0))
            [A,C,U]=hmflip(A,C,LC,LR,U,l,r);  % 详解: 执行语句
            break;  % 详解: 跳出循环：break
        else  % 详解: 条件判断：else 分支
            
            LC(l)=r;  % 详解: 执行语句
            
            SLC=[SLC l];  % 详解: 赋值：计算表达式并保存到 SLC
            
            r=C(l);  % 详解: 赋值：将 C(...) 的结果保存到 r
            
            LR(r)=l;  % 详解: 执行语句
            
            SLR=[SLR r];  % 详解: 赋值：计算表达式并保存到 SLR
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

T=sum(orig(logical(sparse(C,1:size(orig,2),1))));  % 详解: 赋值：将 sum(...) 的结果保存到 T


function A=hminired(A)  % 详解: 执行语句


[m,n]=size(A);  % 详解: 获取向量/矩阵尺寸

colMin=min(A);  % 详解: 赋值：将 min(...) 的结果保存到 colMin
A=A-colMin(ones(n,1),:);  % 详解: 赋值：计算表达式并保存到 A

rowMin=min(A')';  % 详解: 赋值：将 min(...) 的结果保存到 rowMin
A=A-rowMin(:,ones(1,n));  % 详解: 赋值：计算表达式并保存到 A

[i,j]=find(A==0);  % 详解: 执行语句

A(1,n+1)=0;  % 详解: 执行语句
for k=1:n  % 详解: for 循环：迭代变量 k 遍历 1:n
    cols=j(k==i)';  % 赋值：设置变量 cols  % 详解: 赋值：将 j(...) 的结果保存到 cols  % 详解: 赋值：将 j(...) 的结果保存到 cols
    A(k,[n+1 cols])=[-cols 0];  % 详解: 执行语句
end  % 详解: 执行语句


function [A,C,U]=hminiass(A)  % 详解: 函数定义：hminiass(A), 返回：A,C,U


[n,np1]=size(A);  % 详解: 获取向量/矩阵尺寸

C=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 C
U=zeros(1,n+1);  % 详解: 赋值：将 zeros(...) 的结果保存到 U

LZ=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 LZ
NZ=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 NZ

for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
	lj=n+1;  % 详解: 赋值：计算表达式并保存到 lj
	j=-A(i,lj);  % 详解: 赋值：计算表达式并保存到 j

    
	while (C(j)~=0)  % 详解: while 循环：当 ((C(j)~=0)) 为真时迭代
		lj=j;  % 详解: 赋值：计算表达式并保存到 lj
		j=-A(i,lj);  % 详解: 赋值：计算表达式并保存到 j
	
		if (j==0)  % 详解: 条件判断：if ((j==0))
			break;  % 详解: 跳出循环：break
		end  % 详解: 执行语句
	end  % 详解: 执行语句

	if (j~=0)  % 详解: 条件判断：if ((j~=0))
		
		C(j)=i;  % 详解: 执行语句
		
		A(i,lj)=A(i,j);  % 详解: 调用函数：A(i,lj)=A(i,j)

		NZ(i)=-A(i,j);  % 详解: 调用函数：NZ(i)=-A(i,j)
		LZ(i)=lj;  % 详解: 执行语句

		A(i,j)=0;  % 详解: 执行语句
	else  % 详解: 条件判断：else 分支


		lj=n+1;  % 详解: 赋值：计算表达式并保存到 lj
		j=-A(i,lj);  % 详解: 赋值：计算表达式并保存到 j
		
		while (j~=0)  % 详解: while 循环：当 ((j~=0)) 为真时迭代
			r=C(j);  % 详解: 赋值：将 C(...) 的结果保存到 r
			
			lm=LZ(r);  % 详解: 赋值：将 LZ(...) 的结果保存到 lm
			m=NZ(r);  % 详解: 赋值：将 NZ(...) 的结果保存到 m
			
			while (m~=0)  % 详解: while 循环：当 ((m~=0)) 为真时迭代
				if (C(m)==0)  % 详解: 条件判断：if ((C(m)==0))
					break;  % 详解: 跳出循环：break
				end  % 详解: 执行语句
				
				lm=m;  % 详解: 赋值：计算表达式并保存到 lm
				m=-A(r,lm);  % 详解: 赋值：计算表达式并保存到 m
			end  % 详解: 执行语句
			
			if (m==0)  % 详解: 条件判断：if ((m==0))
				lj=j;  % 详解: 赋值：计算表达式并保存到 lj
				j=-A(i,lj);  % 详解: 赋值：计算表达式并保存到 j
			else  % 详解: 条件判断：else 分支
			
				A(r,lm)=-j;  % 详解: 执行语句
				A(r,j)=A(r,m);  % 详解: 调用函数：A(r,j)=A(r,m)
			
				NZ(r)=-A(r,m);  % 详解: 调用函数：NZ(r)=-A(r,m)
				LZ(r)=j;  % 详解: 执行语句
			
				A(r,m)=0;  % 详解: 执行语句
			
				C(m)=r;  % 详解: 执行语句
			
				A(i,lj)=A(i,j);  % 详解: 调用函数：A(i,lj)=A(i,j)
			
				NZ(i)=-A(i,j);  % 详解: 调用函数：NZ(i)=-A(i,j)
				LZ(i)=lj;  % 详解: 执行语句
			
				A(i,j)=0;  % 详解: 执行语句
			
				C(j)=i;  % 详解: 执行语句
				
				break;  % 详解: 跳出循环：break
			end  % 详解: 执行语句
		end  % 详解: 执行语句
	end  % 详解: 执行语句
end  % 详解: 执行语句


r=zeros(1,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 r
rows=C(C~=0);  % 详解: 赋值：将 C(...) 的结果保存到 rows
r(rows)=rows;  % 详解: 执行语句
empty=find(r==0);  % 详解: 赋值：将 find(...) 的结果保存到 empty

U=zeros(1,n+1);  % 详解: 赋值：将 zeros(...) 的结果保存到 U
U([n+1 empty])=[empty 0];  % 详解: 执行语句


function [A,C,U]=hmflip(A,C,LC,LR,U,l,r)  % 详解: 函数定义：hmflip(A,C,LC,LR,U,l,r), 返回：A,C,U


n=size(A,1);  % 详解: 赋值：将 size(...) 的结果保存到 n

while (1)  % 详解: while 循环：当 ((1)) 为真时迭代
    C(l)=r;  % 详解: 执行语句
    
    
    m=find(A(r,:)==-l);  % 详解: 赋值：将 find(...) 的结果保存到 m
    
    A(r,m)=A(r,l);  % 详解: 调用函数：A(r,m)=A(r,l)
    
    A(r,l)=0;  % 详解: 执行语句
    
    if (LR(r)<0)  % 详解: 条件判断：if ((LR(r)<0))
        ...remove row from unassigned row list and return.  % 详解: 执行语句
        U(n+1)=U(r);  % 详解: 调用函数：U(n+1)=U(r)
        U(r)=0;  % 详解: 执行语句
        return;  % 详解: 返回：从当前函数返回
    else  % 详解: 条件判断：else 分支
        
        l=LR(r);  % 详解: 赋值：将 LR(...) 的结果保存到 l
        
        A(r,l)=A(r,n+1);  % 详解: 调用函数：A(r,l)=A(r,n+1)
        A(r,n+1)=-l;  % 详解: 执行语句
        
        r=LC(l);  % 详解: 赋值：将 LC(...) 的结果保存到 r
    end  % 详解: 执行语句
end  % 详解: 执行语句


function [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)  % 详解: 函数定义：hmreduce(A,CH,RH,LC,LR,SLC,SLR), 返回：A,CH,RH


n=size(A,1);  % 详解: 赋值：将 size(...) 的结果保存到 n

coveredRows=LR==0;  % 详解: 赋值：计算表达式并保存到 coveredRows

coveredCols=LC~=0;  % 详解: 赋值：计算表达式并保存到 coveredCols

r=find(~coveredRows);  % 详解: 赋值：将 find(...) 的结果保存到 r
c=find(~coveredCols);  % 详解: 赋值：将 find(...) 的结果保存到 c

m=min(min(A(r,c)));  % 详解: 赋值：将 min(...) 的结果保存到 m

A(r,c)=A(r,c)-m;  % 详解: 执行语句

for j=c  % 详解: for 循环：迭代变量 j 遍历 c
    for i=SLR  % 详解: for 循环：迭代变量 i 遍历 SLR
        if (A(i,j)==0)  % 详解: 条件判断：if ((A(i,j)==0))
            if (RH(i)==0)  % 详解: 条件判断：if ((RH(i)==0))
                RH(i)=RH(n+1);  % 详解: 调用函数：RH(i)=RH(n+1)
                RH(n+1)=i;  % 详解: 执行语句
                CH(i)=j;  % 详解: 执行语句
            end  % 详解: 执行语句
            row=A(i,:);  % 详解: 赋值：将 A(...) 的结果保存到 row
            colsInList=-row(row<0);  % 详解: 赋值：计算表达式并保存到 colsInList
            if (length(colsInList)==0)  % 详解: 条件判断：if ((length(colsInList)==0))
                l=n+1;  % 详解: 赋值：计算表达式并保存到 l
            else  % 详解: 条件判断：else 分支
                l=colsInList(row(colsInList)==0);  % 详解: 赋值：将 colsInList(...) 的结果保存到 l
            end  % 详解: 执行语句
            A(i,l)=-j;  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

r=find(coveredRows);  % 详解: 赋值：将 find(...) 的结果保存到 r
c=find(coveredCols);  % 详解: 赋值：将 find(...) 的结果保存到 c

[i,j]=find(A(r,c)<=0);  % 详解: 执行语句

i=r(i);  % 详解: 赋值：将 r(...) 的结果保存到 i
j=c(j);  % 详解: 赋值：将 c(...) 的结果保存到 j

for k=1:length(i)  % 详解: for 循环：迭代变量 k 遍历 1:length(i)
    lj=find(A(i(k),:)==-j(k));  % 详解: 赋值：将 find(...) 的结果保存到 lj
    A(i(k),lj)=A(i(k),j(k));  % 详解: 调用函数：A(i(k),lj)=A(i(k),j(k))
    A(i(k),j(k))=0;  % 详解: 执行语句
end  % 详解: 执行语句

A(r,c)=A(r,c)+m;  % 详解: 执行语句




