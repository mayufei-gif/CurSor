function x2 =intervaltransfer(qujian,lb,ub,x)
%TOPSIS�����������ͱ����Ĺ淶������qijian��ʾ�����������䣬lb��ʾ�½磬ub��ʾ�Ͻ�
x2=(1-(qujian(1)-x)./(qujian(1)-lb)).*(x>=lb&x<qujian(1))+(x>=qujian(1)&x<=qujian(2))+(1-(x-qujian(2))./(ub-qujian(2))).*(x>qujian(2)&x<=ub);

end

