% 文件: GAbox.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%求解y=variable .* sin(10 * pi * variable) + 2.0的最大值
figure(1);  % 详解: 调用函数：figure(1)
fplot('variable .* sin(10 * pi * variable) + 2.0',[-1, 2]);  % 详解: 调用函数：fplot('variable .* sin(10 * pi * variable) + 2.0',[-1, 2])
NIND = 40;  % 详解: 赋值：计算表达式并保存到 NIND
MAXGEN = 25;  % 详解: 赋值：计算表达式并保存到 MAXGEN
PRECI = 20;  % 详解: 赋值：计算表达式并保存到 PRECI
GGAP = 0.9;  % 详解: 赋值：计算表达式并保存到 GGAP
trace = zeros(2, MAXGEN);  % 详解: 赋值：将 zeros(...) 的结果保存到 trace
FieldD = [20; -1;2;1;0;1;1];  % 详解: 赋值：计算表达式并保存到 FieldD
Chrom = crtbp(NIND,PRECI);  % 详解: 赋值：将 crtbp(...) 的结果保存到 Chrom
gen = 0;  % 详解: 赋值：计算表达式并保存到 gen
variable = bs2rv(Chrom,FieldD);  % 详解: 赋值：将 bs2rv(...) 的结果保存到 variable
ObjV = variable .* sin(10 * pi * variable) + 2.0;  % 详解: 赋值：计算表达式并保存到 ObjV
while gen <= MAXGEN,  % 详解: while 循环：当 (gen <= MAXGEN,) 为真时迭代

    FitnV = ranking(-ObjV);  % 详解: 赋值：将 ranking(...) 的结果保存到 FitnV
SelCh = select('sus',Chrom,FitnV,GGAP);  % 详解: 赋值：将 select(...) 的结果保存到 SelCh
SelCh = recombin('xovsp',SelCh,0.7);  % 详解: 赋值：将 recombin(...) 的结果保存到 SelCh
SelCh = mut(SelCh);  % 详解: 赋值：将 mut(...) 的结果保存到 SelCh
variable = bs2rv(SelCh,FieldD);  % 详解: 赋值：将 bs2rv(...) 的结果保存到 variable
ObjVSel = variable .* sin(10 * pi * variable) + 2.0;  % 详解: 赋值：计算表达式并保存到 ObjVSel
[Chrom ObjV] = reins(Chrom, SelCh, 1, 1, ObjV, ObjVSel);  % 详解: 执行语句
gen = gen + 1;  % 详解: 赋值：计算表达式并保存到 gen
[Y,I] = max(ObjV),hold on;  % 详解: 统计：最大/最小值
plot(variable, Y, 'bo');  % 详解: 调用函数：plot(variable, Y, 'bo')
trace(1, gen) = max(ObjV);  % 详解: 调用函数：trace(1, gen) = max(ObjV)
trace(2, gen) = sum(ObjV) / length(ObjV);  % 详解: 调用函数：trace(2, gen) = sum(ObjV) / length(ObjV)
end  % 详解: 执行语句
variable = bs2rv(Chrom, FieldD);  % 详解: 赋值：将 bs2rv(...) 的结果保存到 variable
hold on,  % 详解: 执行语句
grid;  % 详解: 执行语句
plot(variable, ObjV, 'b*');  % 详解: 调用函数：plot(variable, ObjV, 'b*')
figure(2);  % 详解: 调用函数：figure(2)
plot(trace(1, :));  % 详解: 调用函数：plot(trace(1, :))
hold on  % 详解: 执行语句
plot(trace(2, :),'-.'); grid;  % 详解: 绘图：二维曲线
legend('解的变化', '种群均值的变化');  % 详解: 调用函数：legend('解的变化', '种群均值的变化')



