%% = = =������= = = %%
clc;
clear all;
close all;
%% = = = = = ���������� = = = = = %%
N = 256;  % ͼ���С
I = phantom(N);  % Shepp-Loganͷģ��
delta = pi/180;  % �Ƕ�����
theta = 0:1:179; % ͶӰ�Ƕ�
theta_num = length(theta);
d = 1;
%% = = = = = ����ͶӰ���� = = = = = %%
P = radon(I,theta);
[mm,nn] = size(P);   % ����ͶӰ���ݾ�����С��г���
e = floor((mm-N-1)/2+1)+1;  % ͶӰ���ݵ�Ĭ��ͶӰ����Ϊ floor((size(I)+1)/2)
P = P(e:N+e-1,:);  % ��ȡ����n�����ݣ���ͶӰ���ݽ϶࣬����������
P1 = reshape(P,N,theta_num); 
%% = = = = = �����˲����� = = = = = %%

fh_RL = RLfilter(N,d);
fh_SL = SLfilter(N,d);
%% = = = = = �˲���ͶӰ�ؽ� = = = = = %%
rec = Backprojection(theta_num,N,P1,delta);

rec_RL = RLfilteredbackprojection(theta_num,N,P1,delta,fh_RL);

rec_SL = SLfilteredbackprojection(theta_num,N,P1,delta,fh_SL);

%% = = = = = �����ʾ = = = = = %%
figure;
subplot(2,2,1),imshow(I),xlabel('(a)256x256ͷģ�ͣ�ԭʼͼ��');
subplot(2,2,2),imshow(rec,[]),xlabel('(b)ֱ�ӷ�ͶӰ�ؽ�ͼ��');
subplot(2,2,3),imshow(rec_RL,[]),xlabel('(c)R-L�����˲���ͶӰ�ؽ�ͼ��');
subplot(2,2,4),imshow(rec_SL,[]),xlabel('(d)S-L�����˲���ͶӰ�ؽ�ͼ��');