% 文件: dijkstra.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function dijkstra(N,D,Origem,Destino)  % 详解: 函数定义：dijkstra(N,D,Origem,Destino)

    DA = zeros();  % 详解: 赋值：将 zeros(...) 的结果保存到 DA
    Ant = [];  % 详解: 赋值：计算表达式并保存到 Ant
    ExpA = [];  % 详解: 赋值：计算表达式并保存到 ExpA
    C = Origem;  % 详解: 赋值：计算表达式并保存到 C

    for i = 1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
        ExpA(i) = 0;  % 详解: 执行语句
        DA(i) = 10000;  % 详解: 执行语句
    end  % 详解: 执行语句
    DA(C) = 0;  % 详解: 执行语句
    while (C ~= Destino) && (C ~= 0)  % 详解: while 循环：当 ((C ~= Destino) && (C ~= 0)) 为真时迭代
        for i = 1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
            if (D(C,i)~=0)&&(ExpA(i)==0)  % 详解: 条件判断：if ((D(C,i)~=0)&&(ExpA(i)==0))
                NovaDA = DA(C) + D(C,i);  % 详解: 赋值：将 DA(...) 的结果保存到 NovaDA
                if NovaDA < DA(i)  % 详解: 条件判断：if (NovaDA < DA(i))
                    DA(i) = NovaDA;  % 详解: 执行语句
                    Ant(i) = C;  % 详解: 执行语句
                end  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        ExpA(C) = 1;  % 详解: 执行语句
        Min = 10000;  % 详解: 赋值：计算表达式并保存到 Min
        C = 0;  % 详解: 赋值：计算表达式并保存到 C
        for i = 1:N  % 详解: for 循环：迭代变量 i 遍历 1:N
            if(ExpA(i)==0)&&(DA(i)<Min)  % 详解: 调用函数：if(ExpA(i)==0)&&(DA(i)<Min)
                Min = DA(i);  % 详解: 赋值：将 DA(...) 的结果保存到 Min
                C = i;  % 详解: 赋值：计算表达式并保存到 C
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    if C == Destino  % 详解: 条件判断：if (C == Destino)
        disp('Caminho mais curto \n');  % 详解: 调用函数：disp('Caminho mais curto \n')
        disp(Destino);  % 详解: 调用函数：disp(Destino)
        while C ~= Origem  % 详解: while 循环：当 (C ~= Origem) 为真时迭代
            C = Ant(C);  % 详解: 赋值：将 Ant(...) 的结果保存到 C
            disp(C);  % 详解: 调用函数：disp(C)
        end  % 详解: 执行语句
    else  % 详解: 条件判断：else 分支
        disp('N鉶 existe caminho unindo as duas cidades \n');  % 详解: 调用函数：disp('N鉶 existe caminho unindo as duas cidades \n')
    end  % 详解: 执行语句
end  % 详解: 执行语句



