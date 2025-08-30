%======读取数据======
file='F:\数学建模\授课内容\数据创立结果.xlsx';%文件路径
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

%======数据划分======
% 70%训练集，30%测试集
train_ratio = 0.7;
train_size = round(n * train_ratio);
trainX = x_norm(1:train_size);%训练集x
trainY = y_norm(1:train_size);%训练集y
testX = x_norm(train_size+1:end);%测试集x
testY = y_norm(train_size+1:end);%测试集y

%======神经网络参数======
inputSize  = 1;%输入层大小
hiddenSize = 10;%隐藏层大小  % 修复：将hid denSize改为hiddenSize
outputSize = 1;%输出层大小
maxEpoch   = 5000;%最大迭代次数
learningRate = 0.1;%学习率
goalErr    = 1e-4;%目标误差

%======训练模型======
% 生成随机权重
W1 = randn(hiddenSize,inputSize)-0.5;  %W1为输入层到隐藏层的权重
b1 = randn(hiddenSize,1)-0.5;%b1为隐藏层的偏置
W2 = randn(outputSize,hiddenSize)-0.5;%W2为隐藏层到输出层的权重
b2 = randn(outputSize,1)-0.5;%b2为输出层的偏置

% 存储误差历史
errors = zeros(maxEpoch,1);

% 训练循环
for epoch = 1:maxEpoch
    total_error = 0;
    
    for k = 1:length(trainX)
        xk = trainX(k);
        yk = trainY(k);
        
        % 前向传播
        z1 = W1*xk + b1;%线性组合隐藏层输入
        a1 = 1./(1+exp(-z1));%隐藏层输出(Sigmoid激活函数)
        z2 = W2*a1 + b2;%线性组合输出层输入
        a2 = z2;%输出层输出(线性激活)
        
        % 计算误差
        error = yk - a2;
        total_error = total_error + 0.5*error^2;
        
        % 反向传播
        delta2 = error;%输出层误差
        delta1 = (W2'*delta2).*a1.*(1-a1);%隐藏层误差
        
        % 更新权重和偏置
        W2 = W2 + learningRate*delta2*a1';%更新隐藏层到输出层的权重
        b2 = b2 + learningRate*delta2;%更新输出层的偏置
        W1 = W1 + learningRate*delta1*xk;%更新输入层到隐藏层的权重
        b1 = b1 + learningRate*delta1;%更新隐藏层的偏置
    end
    
    errors(epoch) = total_error;
    
    % 输出误差
    if mod(epoch,500)==0 || total_error < goalErr
        fprintf('epoch=%d, 误差=%.6f\n',epoch,total_error);
    end
    
    if total_error < goalErr
        fprintf('达到目标误差，训练完成！\n');
        break;
    end
end

%======模型评估======
% 训练集预测
train_pred_norm = predict(W1, b1, W2, b2, trainX);
test_pred_norm = predict(W1, b1, W2, b2, testX);

% 反归一化
train_pred = train_pred_norm * (ymax - ymin) + ymin;
test_pred = test_pred_norm * (ymax - ymin) + ymin;
trainY_actual = trainY * (ymax - ymin) + ymin;
testY_actual = testY * (ymax - ymin) + ymin;

% 计算误差指标
train_mse = mean((train_pred - trainY_actual).^2);
test_mse = mean((test_pred - testY_actual).^2);
train_rmse = sqrt(train_mse);
test_rmse = sqrt(test_mse);

fprintf('训练集RMSE: %.4f\n', train_rmse);
fprintf('测试集RMSE: %.4f\n', test_rmse);

%======可视化结果======
% 反归一化所有预测值
all_pred_norm = predict(W1, b1, W2, b2, x_norm);
all_pred = all_pred_norm * (ymax - ymin) + ymin;

% 绘制结果
figure('Position', [100, 100, 1200, 400]);

% 子图1：原始数据与拟合曲线
subplot(1,3,1);
plot(x, y, 'bo', 'MarkerSize', 6, 'DisplayName', '原始数据');
hold on;
plot(x, all_pred, 'r-', 'LineWidth', 2, 'DisplayName', '神经网络拟合');
xlabel('x');
ylabel('y');
title('神经网络拟合结果');
legend;
grid on;

% 子图2：训练误差曲线
subplot(1,3,2);
plot(1:epoch, errors(1:epoch), 'b-', 'LineWidth', 2);
xlabel('迭代次数');
ylabel('训练误差');
title('训练误差曲线');
grid on;

% 子图3：测试集预测结果
subplot(1,3,3);
plot(testY_actual, 'bo', 'MarkerSize', 6, 'DisplayName', '实际值');
hold on;
plot(test_pred, 'rx', 'MarkerSize', 6, 'DisplayName', '预测值');
xlabel('样本序号');
ylabel('y值');
title('测试集预测结果');
legend;
grid on;

% 保存训练好的模型
save('neural_network_model.mat', 'W1', 'b1', 'W2', 'b2', 'xmin', 'xmax', 'ymin', 'ymax');

% 使用模型进行新数据预测
fprintf('神经网络训练完成！模型已保存到 neural_network_model.mat\n');

%======预测函数======
function predictions = predict(W1, b1, W2, b2, X)
predictions = zeros(size(X));
for i = 1:length(X)
    % 前向传播
    z1 = W1*X(i) + b1;
    a1 = 1./(1+exp(-z1));
    z2 = W2*a1 + b2;
    predictions(i) = z2;
end
end
