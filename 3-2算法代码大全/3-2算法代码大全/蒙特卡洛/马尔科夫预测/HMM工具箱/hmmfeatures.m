% 文件: hmmfeatures.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function y=hmmfeatures(s,N,deltaN,M,Q)  % 详解: 执行语句
Ns=length(s);  % 详解: 赋值：将 length(...) 的结果保存到 Ns
T=1+fix((Ns-N)/deltaN);  % 详解: 赋值：计算表达式并保存到 T
a=zeros(Q,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 a
gamma=zeros(Q,1);  % 详解: 赋值：将 zeros(...) 的结果保存到 gamma
gamma_w=zeros(Q,T);  % 详解: 赋值：将 zeros(...) 的结果保存到 gamma_w
win_gamma=1+(Q/2)*sin(pi/Q*(1:Q)');  % 赋值：设置变量 win_gamma  % 详解: 赋值：计算表达式并保存到 win_gamma  % 详解: 赋值：计算表达式并保存到 win_gamma
for (t=1:T)  % 详解: 调用函数：for(t=1:T)
    idx=(deltaN*(t-1)+1):(deltaN*(t-1)+N);  % 详解: 赋值：计算表达式并保存到 idx
    sw=s(idx).*hamming(N);  % 详解: 赋值：将 s(...) 的结果保存到 sw
    [rs,eta]=xcorr(sw,M,'biased');  % 详解: 执行语句
   a=levinson(rs(M+1:2*M+1),M);  % 详解: 赋值：将 levinson(...) 的结果保存到 a
    a=a(2:M+1)';  % 赋值：设置变量 a  % 详解: 赋值：将 a(...) 的结果保存到 a  % 详解: 赋值：将 a(...) 的结果保存到 a
    gamma(1)=a(1);  % 详解: 调用函数：gamma(1)=a(1)
     for (i=2:Q)  % 详解: 调用函数：for(i=2:Q)
         gamma(i)=a(i)+(1:i-1)*(gamma(1:i-1).*a(i-1:-1:1))/i;  % 详解: 执行语句
     end  % 详解: 执行语句
     gamma_w(:,t)=gamma.*win_gamma;  % 详解: 执行语句
end  % 详解: 执行语句
detla_gamma_w=gradient(gamma_w);  % 详解: 赋值：将 gradient(...) 的结果保存到 detla_gamma_w
y=[gamma_w;detla_gamma_w];  % 详解: 赋值：计算表达式并保存到 y
         
         
         
         
         
         



