clc;
clear all;
close all;
N = 256; %ͼ���С
I = phantom(N); %shepp-loganͷģ��
theta = 0:1:179; %ͶӰ�Ƕ�
P = radon(I,theta); %����ͶӰ����
rc = iradon(P,theta,'linear','None');%ֱ�ӷ�ͶӰ�ؽ�
figure;%��ʾͼ��
imshow(I),title('ԭʼͼ��');
figure;
imshow(rc,[]),title('ֱ�ӷ�ͶӰ�ؽ�ͼ��');
