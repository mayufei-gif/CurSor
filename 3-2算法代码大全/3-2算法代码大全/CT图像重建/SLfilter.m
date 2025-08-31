function fh_SL = SLfilter(N,d)
% S-L filter function
% ----------------------
% ���������
% N:ͼ���С��̽����ͨ��������
% d:ƽ�Ʋ���
% ���������
%fh_SL:S-L�˲�����
fh_SL = zeros(1,N);
for k1 = 1:N
    fh_SL(k1) = -2/(pi^2*d^2*(4*(k1-N/2-1)^2-1));
end

