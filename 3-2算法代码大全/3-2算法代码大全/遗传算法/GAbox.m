%���y=variable .* sin(10 * pi * variable) + 2.0�����ֵ
figure(1);
fplot('variable .* sin(10 * pi * variable) + 2.0',[-1, 2]);%����Ŀ�꺯��ͼ
%�����Ŵ��㷨����
NIND = 40;%������Ŀ
MAXGEN = 25;%����Ŵ�����
PRECI = 20;%�����Ķ�����λ����ע��Ҳ���Ǿ��ȣ�
GGAP = 0.9;%����
trace = zeros(2, MAXGEN);%Ѱ�Ž������ʼֵ
FieldD = [20; -1;2;1;0;1;1];%����������
Chrom = crtbp(NIND,PRECI);%��ʼ��Ⱥ
gen = 0;%��������
variable = bs2rv(Chrom,FieldD);%��ʼ��Ⱥת��ʮ������
ObjV = variable .* sin(10 * pi * variable) + 2.0;%Ŀ�꺯��ֵ
while gen <= MAXGEN,

    FitnV = ranking(-ObjV);%������Ӧ��ֵ
%��ע������ranking�Ĺ����ǣ������������Ӧ��ֵ���䣬���ݸ����Ŀ��ֵ��С�����˳������ǽ�������
%������һ������Ӧ������Ӧ��ֵFitnV����������
%ѹ����ָ������ĸ�����Ӧ��ֵ�Ĳ�࣬��õĺ���ģ�����ԭ���ĺ���ֵû��Ӱ�죬ֻ�Ǹ���ԭ���ĺ���ֵ������һ�����ָ������ӵ�һ��ֵ����
SelCh = select('sus',Chrom,FitnV,GGAP);%ѡ��
SelCh = recombin('xovsp',SelCh,0.7);%���飬������
SelCh = mut(SelCh); %����
variable = bs2rv(SelCh,FieldD);%�Ӵ�ʮ����ת��
ObjVSel = variable .* sin(10 * pi * variable) + 2.0;%�Ӵ�Ŀ�꺯��ֵ
[Chrom ObjV] = reins(Chrom, SelCh, 1, 1, ObjV, ObjVSel);%�ز����Ӵ�����Ⱥ
gen = gen + 1; %(ע��matlab��֧��gen += 1��
[Y,I] = max(ObjV),hold on;
plot(variable, Y, 'bo');
trace(1, gen) = max(ObjV); %�Ŵ��㷨���ܸ���
trace(2, gen) = sum(ObjV) / length(ObjV);
end
variable = bs2rv(Chrom, FieldD);%���Ÿ����ʮ����ת��
hold on,
grid;
plot(variable, ObjV, 'b*');
figure(2);
plot(trace(1, :));
hold on
plot(trace(2, :),'-.'); grid;
legend('��ı仯', '��Ⱥ��ֵ�ı仯');