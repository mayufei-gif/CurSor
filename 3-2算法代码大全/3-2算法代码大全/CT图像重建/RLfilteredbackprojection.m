% 文件: RLfilteredbackprojection.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function rec_RL = RLfilteredbackprojection(theta_num,N,R1,delta,fh_RL)  % 详解: 执行语句

rec_RL = zeros(N);  % 详解: 赋值：将 zeros(...) 的结果保存到 rec_RL
for m = 1:theta_num  % 详解: for 循环：迭代变量 m 遍历 1:theta_num
    pm = R1(:,m);  % 详解: 赋值：将 R1(...) 的结果保存到 pm
    pm_RL = conv(fh_RL,pm,'same');  % 详解: 赋值：将 conv(...) 的结果保存到 pm_RL
    Cm = (N/2)*(1-cos((m-1)*delta)-sin((m-1)*delta));  % 详解: 赋值：计算表达式并保存到 Cm
    for k1 = 1:N  % 详解: for 循环：迭代变量 k1 遍历 1:N
        for k2 = 1:N  % 详解: for 循环：迭代变量 k2 遍历 1:N
            Xrm = Cm+(k2-1)*cos((m-1)*delta)+(k1-1)*sin((m-1)*delta);  % 详解: 赋值：计算表达式并保存到 Xrm
            n = floor(Xrm);  % 详解: 赋值：将 floor(...) 的结果保存到 n
            t = Xrm-floor(Xrm);  % 详解: 赋值：计算表达式并保存到 t
            n = max(1,n);n = min(n,N-1);  % 详解: 赋值：将 max(...) 的结果保存到 n
            p_RL = (1-t)*pm_RL(n) + t*pm_RL(n+1);  % 详解: 赋值：计算表达式并保存到 p_RL
            rec_RL(N+1-k1,k2) = rec_RL(N+1-k1,k2)+p_RL;  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句

            
            
    
    




