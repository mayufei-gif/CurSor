% 文件: ParallelBeam.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function P=ParallelBeam(theta, N, P_num)  % 详解: 执行语句

shep = [1    .69    .92     0     0       0  % 详解: 赋值：计算表达式并保存到 shep
        -.8  .6624  .8740   0     -.0184  0  % 详解: 执行语句
        -.2  .1100  .3100   .22   0       -18  % 详解: 执行语句
        -.2  .1600  .4100   -.22  0       18  % 详解: 执行语句
        .1   .2100  .2500   0     .35     0  % 详解: 执行语句
        .1   .0460  .0460   0     .1      0  % 详解: 执行语句
        .1   .0460  .0460   0     -.1     0  % 详解: 执行语句
        .1   .0460  .0230   -.08  -.605   0  % 详解: 执行语句
        .1   .0230  .0230   0     -.606   0  % 详解: 执行语句
        .1   .0230  .0460   .06   -.605   0];  % 详解: 执行语句
    theta_num = length(theta);  % 详解: 赋值：将 length(...) 的结果保存到 theta_num
    P = zeros(P_num ,theta_num);  % 详解: 赋值：将 zeros(...) 的结果保存到 P
    rho = shep(:,1).';  % 赋值：设置变量 rho  % 详解: 赋值：将 shep(...) 的结果保存到 rho  % 详解: 赋值：将 shep(...) 的结果保存到 rho
    ae = 0.5 * N * shep(:,2).';  % 赋值：设置变量 ae  % 详解: 赋值：计算表达式并保存到 ae  % 详解: 赋值：计算表达式并保存到 ae
    be = 0.5 * N * shep(:,3).';  % 赋值：设置变量 be  % 详解: 赋值：计算表达式并保存到 be  % 详解: 赋值：计算表达式并保存到 be
    xe = 0.5 * N * shep(:,4).';  % 赋值：设置变量 xe  % 详解: 赋值：计算表达式并保存到 xe  % 详解: 赋值：计算表达式并保存到 xe
    ye = 0.5 * N * shep(:,5).';  % 赋值：设置变量 ye  % 详解: 赋值：计算表达式并保存到 ye  % 详解: 赋值：计算表达式并保存到 ye
    alpha = shep(:,6).';  % 赋值：设置变量 alpha  % 详解: 赋值：将 shep(...) 的结果保存到 alpha  % 详解: 赋值：将 shep(...) 的结果保存到 alpha
    alpha = alpha * pi/180;  % 详解: 赋值：计算表达式并保存到 alpha
    theta = theta * pi/180;  % 详解: 赋值：计算表达式并保存到 theta
    TT = -(P_num-1)/2:(P_num-1)/2;  % 详解: 赋值：计算表达式并保存到 TT
    for k1 = 1:theta_num  % 详解: for 循环：迭代变量 k1 遍历 1:theta_num
        P_theta = zeros(1,P_num);  % 详解: 赋值：将 zeros(...) 的结果保存到 P_theta
        for k2 = 1:max(size(xe))  % 详解: for 循环：迭代变量 k2 遍历 1:max(size(xe))
            a = (ae(k2) * cos(theta(k1)- alpha(k2)))^2+ (be(k2)* sin(theta(k1)- alpha(k2)))^2;  % 详解: 赋值：计算表达式并保存到 a
            temp = a -(TT - xe(k2)* cos(theta(k1))- ye(k2)* sin(theta(k1))).^2;  % 详解: 赋值：计算表达式并保存到 temp
            ind = temp >0;  % 详解: 赋值：计算表达式并保存到 ind
            P_theta(ind) = P_theta(ind) +rho(k2) *(2 *ae(k2)* be(k2)* sqrt(temp(ind)))./a;  % 详解: 执行语句
        end  % 详解: 执行语句
        P(:,k1) = P_theta.';  % 调用函数：P  % 详解: 执行语句  % 详解: 执行语句
    end  % 详解: 执行语句
    



