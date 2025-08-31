clc,clear 
load berlin52.tsp %���صз�52��Ŀ������ݣ����ݰ��ձ���е�λ�ñ����ڴ��ı��ļ�sj.txt��
berlin52(:,1)=[];
x=berlin52(:,1);
y=berlin52(:,2);
%�������d 
d=zeros(52); 
for i=1:52
for j=1:52
d(i,j)=sqrt((x(i)-x(j)).^2+(y(i)-y(j)).^2);
end 
end

figure(1)
plot(x,y,'o')
hold on

S0=[];Sum=inf; 
rand('state',sum(clock)); 
for j=1:1000 
S=[1 1+randperm(51),52]; 
temp=0;
for i=1:52
temp=temp+d(S(i),S(i+1)); 
end 
if temp<Sum 
S0=S;Sum=temp; 
end 
end 
e=0.1^30;L=2000;at=0.999;T=1; 
%�˻����
for k=1:L 
    k
%�����½�
c=2+floor(50*rand(2,1)); 
c=sort(c); 
c1=c(1);c2=c(2); 
%������ۺ���ֵ
df=d(S0(c1-1),S0(c2))+d(S0(c1),S0(c2+1))-d(S0(c1-1),S0(c1))-d(S0(c2),S0(c2+1)); 
%����׼��
if df<0 
S0=[S0(1:c1-1),S0(c2:-1:c1),S0(c2+1:53)]; 
Sum=Sum+df; 
elseif exp(-df/T)>rand(1) 
S0=[S0(1:c1-1),S0(c2:-1:c1),S0(c2+1:53)]; 
Sum=Sum+df; 
end 
T=T*at; 
if T<e 
break; 
end 
plot(x(S0),y(S0),'--')
hold on
end 
% ���Ѳ��·����·������
S0,Sum 
figure(2)
plot(x,y,'*',x(S0),y(S0),'-')

