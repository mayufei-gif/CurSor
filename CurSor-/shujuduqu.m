% [xdata,txtdata]=xlsread(""); %读取文件路径下的excel文件
% xdata=csvread('');%读取文件路径下的csv文件
% xdata=readtable('');%读取文件路径下的txt文件
% xdata=readtable('');%读取文件路径下的xlsx文件
% xdata=readtable('');%读取文件路径下的xls文件
% xdata=readtable('');%读取文件路径下的mat文件

%======读取数据======
file='C:\Users\10101\Desktop\shuju.xlsx';%文件路径
M  = readmatrix(file);          % 读取excel文件
odd  = M(:,1:2:end);                   % 奇列
even = M(:,2:2:end);                   % 偶列
xdata=[odd,even];%将奇数列和偶数列合并为新的矩阵
writematrix(xdata,'new.xlsx');          % 写入excel文件
%======数据处理======
[n,m]=size(xdata);%n为数据行数，m为数据列数
x=xdata(:,1);%取第一列数据为x
y=xdata(:,2);%取第二列数据为y
xmin = min(x); xmax = max(x);%x的最小值和最大值
ymin = min(y); ymax = max(y);%y的最小值和最大值
x_norm = (x - xmin)/(xmax - xmin);%归一化x
y_norm = (y - ymin)/(ymax - ymin);%归一化y

% normParam.x = [xmin xmax];%归一化x的范围
% normParam.y = [ymin ymax];%归一化y的范围
% save('normParam.mat','normParam');%保存归一化参数

trainX = x_norm(1:20);%训练集x
trainY = y_norm(1:20);%训练集y
inputSize  = 1;%输入层大小
hiddenSize = 10;%隐藏层大小
outputSize = 1;%输出层大小
maxEpoch   = 5000;%最大迭代次数
learningRate = 0.1;%学习率
goalErr    = 1e-4;%目标误差

%======训练模型======
% 生成随机权重
W1 = randn(hiddenSize,outputSize)-0.5;  %W1为隐藏层到输入层的权重
b1 = randn(hiddenSize,outputSize)-0.5;%b1为隐藏层的偏置
W2 = randn(outputSize,hiddenSize);%W2为输出层到隐藏层的权重
b2 = randn(outputSize,1);%b2为输出层的偏置
% 增量规则
for epoch = 1:maxEpoch  %迭代次数
    E = 0;  %误差
    for k = 1:20  %训练集数据
        xk = trainX(k);%训练集x
        yk = trainY(k);%训练集y
        
        % 前向传播
        z1 = W1*xk + b1;%线性组合隐藏层输入
        a1 = 1./(1+exp(-z1));%隐藏层输出
        z2 = W2*a1 + b2;%线性组合输出层输入
        a2 = z2;%输出层输出
        % 计算误差
        E = E + 0.5*(yk-a2)^2;%计算误差
        % 反向传播
        delta2 = (yk-a2);%输出层误差
        delta1 = (W2'*delta2).*a1.*(1-a1);%隐藏层误差
        % 更新权重和偏置
        W2 = W2 + learningRate*delta2*a1';%更新输出层到隐藏层的权重
        b2 = b2 + learningRate*delta2;%更新输出层的偏置
        W1 = W1 + learningRate*delta1*xk';%更新隐藏层到输入层的权重
        b1 = b1 + learningRate*delta1;%更新隐藏层的偏置
    end
    % 输出误差
    if mod(epoch,500)==0 || E < goalErr%每500次迭代或误差小于目标误差时输出一次
        fprintf('epoch=%d, E=%f\n',epoch,E);%输出迭代次数和误差
    end
    if E < goalErr, break; %如果误差小于目标误差，跳出循环
    end
end






% 计算误差




% % 初始化误差
% err = Inf;%初始化误差
% iter = 0;%初始化迭代次数

% % 画图
% figure(1);%新建一个图形窗口
% plot(x_norm,y_norm,'r*');%画图
% xlabel('x');%x轴标签
% ylabel('y');%y轴标签
% title('归一化后散点图');%标题
% % 画图
% figure(2);%新建一个图形窗口
% plot(x_norm,y_norm,'r*');%画图
% xlabel('x');%x轴标签
% ylabel('y');%y轴标签
% title('归一化后散点图');%标题
% % 画图
% figure(3);%新建一个图形窗口
% plot(x_norm,y_norm,'r*');%画图
% xlabel('x');%x轴标签
% ylabel('y');%y轴标签
% title('归一化后散点图');%标题


% [Output]
