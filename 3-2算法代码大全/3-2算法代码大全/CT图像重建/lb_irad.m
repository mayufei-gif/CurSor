clc;
clear all;
close all;
N = 256; %ͼ���С
I = phantom(N); %shepp-loganͷģ��
theta = 0:1:179; %ͶӰ�Ƕ�
P = radon(I,theta); %����ͶӰ����
rc = iradon(P,theta,'linear','None');%ֱ�ӷ�ͶӰ�ؽ�
rec_RL = iradon(P,theta,'linear','Ram-Lak');% Ĭ���˲���
rec_SL = iradon(P,theta,'linear','Shepp-Logan');
figure;%��ʾͼ��
subplot(2,2,1),imshow(I),title('ԭʼͼ��');
subplot(2,2,2),imshow(rc,[]),title('ֱ�ӷ�ͶӰ�ؽ�ͼ��');
subplot(2,2,3),imshow(rec_RL,[]),title('R-L�����˲���ͶӰ�ؽ�ͼ��');
subplot(2,2,4),imshow(rec_SL,[]),title('S-L�����˲���ͶӰ�ؽ�ͼ��');

