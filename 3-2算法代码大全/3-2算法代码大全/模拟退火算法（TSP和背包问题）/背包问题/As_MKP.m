% 文件: As_MKP.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%模拟退火算法解决背包问题
%0-1背包，假设12件物品，质量分别为2磅，5磅，18，3，2，5，10，4，11，7，14，6磅
%价值分别为5元，10，13，4，3，11，13，10，8，16，7，4元
%包的最大允许质量为46磅
clear  % 详解: 执行语句
clc  % 详解: 执行语句
a=0.95;  % 详解: 赋值：计算表达式并保存到 a
k=[5;10;13;4;3;11;13;10;8;16;7;4];  % 详解: 赋值：计算表达式并保存到 k
k=-k;  % 详解: 赋值：计算表达式并保存到 k
d=[2;5;18;3;2;5;10;4;11;7;14;6];  % 详解: 赋值：计算表达式并保存到 d
restriction=46;  % 详解: 赋值：计算表达式并保存到 restriction
num=12;  % 详解: 赋值：计算表达式并保存到 num
sol_new=ones(1,num);  % 详解: 赋值：将 ones(...) 的结果保存到 sol_new
E_current=inf;E_best=inf;  % 详解: 赋值：计算表达式并保存到 E_current
sol_current=sol_new;sol_best=sol_new;  % 详解: 赋值：计算表达式并保存到 sol_current
t0=97; tf=3;  t=t0;  % 详解: 赋值：计算表达式并保存到 t0
p=1;  % 详解: 赋值：计算表达式并保存到 p

while t>=tf  % 详解: while 循环：当 (t>=tf) 为真时迭代
    for r=1:100  % 详解: for 循环：迭代变量 r 遍历 1:100
        tmp=ceil(rand.*num);  % 详解: 赋值：将 ceil(...) 的结果保存到 tmp
        sol_new(1,tmp)=~sol_new(1,tmp);  % 详解: 调用函数：sol_new(1,tmp)=~sol_new(1,tmp)
        while 1  % 详解: while 循环：当 (1) 为真时迭代
            q=(sol_new*d<=restriction);  % 详解: 赋值：计算表达式并保存到 q
            if~q  % 详解: 执行语句
                p=~p  % 详解: 赋值：计算表达式并保存到 p
                tmp=find(sol_new==1);  % 详解: 赋值：将 find(...) 的结果保存到 tmp
                if p  % 详解: 条件判断：if (p)
                     sol_new(1,tmp)=0;  % 详解: 执行语句
                else  % 详解: 条件判断：else 分支
                    sol_new(1,tmp(end))=0;  % 详解: 执行语句
                end  % 详解: 执行语句
            else  % 详解: 条件判断：else 分支
                break  % 详解: 跳出循环：break
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        
E_new=sol_new*k;  % 详解: 赋值：计算表达式并保存到 E_new
if E_new<E_current  % 详解: 条件判断：if (E_new<E_current)
   E_current=E_new;  % 详解: 赋值：计算表达式并保存到 E_current
   sol_current=sol_new;  % 详解: 赋值：计算表达式并保存到 sol_current
   if E_new<E_best  % 详解: 条件判断：if (E_new<E_best)
       E_best=E_new;  % 详解: 赋值：计算表达式并保存到 E_best
       sol_best=sol_new;  % 详解: 赋值：计算表达式并保存到 sol_best
   end  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
    if rand<exp(-(E_new-E_current)./t)  % 详解: 条件判断：if (rand<exp(-(E_new-E_current)./t))
        E_current=E_new;  % 详解: 赋值：计算表达式并保存到 E_current
        sol_current=sol_new;  % 详解: 赋值：计算表达式并保存到 sol_current
    else  % 详解: 条件判断：else 分支
        sol_new=sol_current;  % 详解: 赋值：计算表达式并保存到 sol_new
    end  % 详解: 执行语句
end  % 详解: 执行语句
    end  % 详解: 执行语句
    t=t.*a;  % 详解: 赋值：计算表达式并保存到 t
end  % 详解: 执行语句

disp('最优解为：')  % 详解: 调用函数：disp('最优解为：')
sol_best;  % 详解: 执行语句
disp('物品总价值等于：')  % 详解: 调用函数：disp('物品总价值等于：')
val= -E_best;  % 详解: 赋值：计算表达式并保存到 val
disp(val)  % 详解: 调用函数：disp(val)
disp('背包中的物品重量为：')  % 详解: 调用函数：disp('背包中的物品重量为：')
disp(sol_best * d)  % 详解: 调用函数：disp(sol_best * d)





