%��У�о���Ժ��TOPSIS����
%�μ�����

clear 
a=[0.1 5 5000 4.7
   0.2 6 6000 5.6];
[m,n]=size(a);
qujian=[5 ,6];lb=2;ub=12;%�������䣬�½磬�Ͻ�
a(:,2)=intervaltransfer(qujian,lb,ub,a(:,2));
%��������ָ��淶��
for j=1:n
    b(:,j)=a(:,j)/norm(a(:,j));
end
w=[0.2,0.3,0.4,0.1];  %��������ָ���Ȩ��
c=b.*repmat(w,m,1);%���Ȩ����
cstar=max(c);%��������⣨���Ч���ͱ�����
cstar(4)=min(c(:,4));%�������4���ɱ���
c0=min(c);%�������
c0(4)=max(c(:,4));%����4Ϊ�ɱ���
for i=1:m
    sstar(i)=norm(c(i,:)-cstar);  %���������ľ���
    s0(i)=norm(c(i,:)-c0);
end
f=s0./(sstar+s0);
[sf,ind]=sort(f,'descend');%��������
sf

