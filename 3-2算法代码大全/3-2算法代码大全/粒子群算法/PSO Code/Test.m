clear;
results=0;
M=50 ;  % �����������������50��
his=zeros((M+2),1);

for i=1:M
    his(i)=PSO();
    results = results+ his(i);
end
avg= results/M  
Std=std(his(1:M))
his=[his',avg,Std]
