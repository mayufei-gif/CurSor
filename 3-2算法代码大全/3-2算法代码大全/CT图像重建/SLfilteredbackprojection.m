function rec_SL = SLfilteredbackprojection(theta_num,N,R1,delta,fh_SL)
% S-L filtered back projection function
% ------------------------------------
% theta_num:ͶӰ�Ƕȸ���
% N:ͼ���С��̽����ͨ������
% R1��ͶӰ���ݾ���
% delta���Ƕ����������ȣ�
% fh_SL:S-L�˲�����
% ���������
% rec_SL:��ͶӰ�ؽ�����
rec_SL = zeros(N);
for m = 1:theta_num
    pm = R1(:,m); % ĳһ�Ƕȵ�ͶӰ����
    pm_SL = conv(fh_SL,pm,'same'); % �����
    Cm = (N/2)*(1-cos((m-1)*delta)-sin((m-1)*delta));
    for k1 = 1:N
        for k2 = 1:N
            % �������������㣬ע���������nȡֵ��ΧΪ1-N-1
            Xrm = Cm+(k2-1)*cos((m-1)*delta)+(k1-1)*sin((m-1)*delta);
            n = floor(Xrm);
            t = Xrm-floor(Xrm); %С������
            n = max(1,n);n = min(n,N-1);
            p_SL = (1-t)*pm_SL(n) + t*pm_SL(n+1); % �����ڲ�
            rec_SL(N+1-k1,k2) = rec_SL(N+1-k1,k2)+p_SL; % ��ͶӰ
        end
    end
end