clear all;close all;clc
%��ʼ���ڽ�ѹ����
b=[1 2;1 3;1 4;2 4;
3 6;4 6;4 7];
c=b(:)
m=max(b(:));             %ѹ���������ֵ�����ڽӾ���Ŀ����
A=compresstable2matrix(b) %���ڽ�ѹ������ͼ�ľ����ʾ
netplot(A,1)                %�����ʾ
title('ԭʼ��������ͼ'); 
top=1;                  %��ջ��
stack(top)=1;           %����һ���ڵ���ջ

flag=1;                 %���ĳ���ڵ��Ƿ���ʹ���
re=[];                  %���ս��
while top~=0            %�ж϶�ջ�Ƿ�Ϊ��
    pre_len=length(stack);    %��Ѱ��һ���ڵ�ǰ�Ķ�ջ����
    i=stack(top);             %ȡ��ջ���ڵ�
    for j=1:m
        if A(i,j)==1 && isempty(find(flag==j,1))    %����ڵ���������û�з��ʹ� 
            top=top+1;                          %��չ��ջ
            stack(top)=j;                       %�½ڵ���ջ
            flag=[flag j];                      %���½ڵ���б��
            re=[re;i j];                        %���ߴ�����
            break;   
        end
    end    
    if length(stack)==pre_len   %�����ջ����û�����ӣ���ڵ㿪ʼ��ջ
        stack(top)=[];
        top=top-1;
    end    
end

A=compresstable2matrix(re);
figure;
netplot(A,1)
title('���������������ͼ'); 