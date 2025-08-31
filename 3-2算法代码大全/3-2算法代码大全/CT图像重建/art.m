clc;
clear all;
close all;
N = 180; %ͼ���С
N2 = N^2;
I = phantom(N);% ����ͷģ��ͼ��
theta = linspace(0,180,181);   %����[0,180]�����Էֲ���181����
theta = theta(1:180);
%% = = = = = = ����ͶӰ���� = = = = = = %%
P_num = 260; %̽����ͨ������
P = ParallelBeam(theta ,N ,P_num); %����ͶӰ����
%P = radon(I,theta);
%% = = = = = = ��ȡͶӰ���� = = = = = =%%
delta = 1;% �����С���Ƕ�����
[W_ind,W_dat] = SystemMatrix(theta,N,P_num,delta);
%% = = = = = = ����ART���� = = = = = = %%
F = zeros(N2,1);  %��ʼͼ������
lambda = 0.25;  %�ɳ�����
c = 0;  %����������
irt_num = 5;  
while(c<irt_num)
    for j = 1:length(theta)
        for i = 1:1:P_num
            % ȡ��һ�������������������źͳ���
            u = W_ind((j-1)*P_num + i,:);  % ���
            v = W_dat((j-1)*P_num + i,:);  % ����
            
            if any(u) == 0
                continue;
            end
            %�ָ�ͶӰ����������һ�����߶�Ӧ�������� w
            w = zeros(1,N2);
            ind = u > 0;
            w(u(ind))=v(ind);
            % ͼ�����һ�ε���
            PP = w * F;   %ǰ��ͶӰ
            C = (P(i,j)-PP)/sum(w.^2) * w';  % ������           
            F = F + lambda * C;
        end
    end
    F(F<0) = 0;   % С��0������ֵ��Ϊ0
    c = c+1;
end
F = reshape(F,N,N)'; % ת����N x N��ͼ�����
%% = = = = = = ��������ʾ = = = = = = %%
figure(1);
imshow(I);xlabel('(a)180x180ͷģ��ͼ��');
figure(2);
%A = imadjust(F);
imshow(F,[]);xlabel('(b)ART�㷨�ؽ���ͼ��');
     
