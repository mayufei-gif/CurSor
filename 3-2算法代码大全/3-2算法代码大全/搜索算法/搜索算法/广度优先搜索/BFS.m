clear all;close all;clc
%��ʼ���ڽ�ѹ����
b=[1 2;1 3;1 4;2 4;3 6;4 6;4 7];

m=max(b(:));                %ѹ���������ֵ�����ڽӾ���Ŀ����
A=compresstable2matrix(b);  %���ڽ�ѹ������ͼ�ľ����ʾ
netplot(A,1)                %�����ʾ

head=1;             %����ͷ
tail=1;             %����β����ʼ����Ϊ�գ�tail==head
queue(head)=1;      %��ͷ�м���ͼ��һ���ڵ�
head=head+1;        %������չ

flag=1;             %���ĳ���ڵ��Ƿ���ʹ���
re=[];              %���ս��
while tail~=head    %�ж϶����Ƿ�Ϊ��
    i=queue(tail);  %ȡ��β�ڵ�
    for j=1:m
        if A(i,j)==1 && isempty(find(flag==j,1))    %����ڵ���������û�з��ʹ�
            queue(head)=j;                          %�½ڵ�����
            head=head+1;                            %��չ����
            flag=[flag j];                          %���½ڵ���б��
            re=[re;i j];                            %���ߴ�����
        end
    end
    tail=tail+1;            
end

A=compresstable2matrix(re);
figure;
netplot(A,1)