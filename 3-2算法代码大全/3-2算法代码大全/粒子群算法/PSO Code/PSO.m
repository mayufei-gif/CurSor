% 文件: PSO.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function fbestval = PSO()  % 详解: 执行语句
fname =@f1;  % 详解: 赋值：计算表达式并保存到 fname
NDim = 30;  % 详解: 赋值：计算表达式并保存到 NDim
MaxIter =3000;  % 详解: 赋值：计算表达式并保存到 MaxIter
Bound = fname();  % 详解: 赋值：将 fname(...) 的结果保存到 Bound





iteration = 0;  % 详解: 赋值：计算表达式并保存到 iteration
PopSize=48;  % 详解: 赋值：计算表达式并保存到 PopSize
w=1;  % 详解: 赋值：计算表达式并保存到 w
c1=2.05;  % 详解: 赋值：计算表达式并保存到 c1
c2=2.05;  % 详解: 赋值：计算表达式并保存到 c2
gbest = zeros(NDim,PopSize);  % 详解: 赋值：将 zeros(...) 的结果保存到 gbest
LowerBound = zeros(NDim,PopSize);  % 详解: 赋值：将 zeros(...) 的结果保存到 LowerBound
UpperBound = zeros(NDim,PopSize);  % 详解: 赋值：将 zeros(...) 的结果保存到 UpperBound
for i=1:PopSize  % 详解: for 循环：迭代变量 i 遍历 1:PopSize
    LowerBound(:,i)=Bound(:,1);  % 详解: 调用函数：LowerBound(:,i)=Bound(:,1)
    UpperBound(:,i)=Bound(:,2);  % 详解: 调用函数：UpperBound(:,i)=Bound(:,2)
end  % 详解: 执行语句

population =  rand(NDim, PopSize).*(UpperBound-LowerBound) + LowerBound;  % 详解: 赋值：将 rand(...) 的结果保存到 population
vmax = ones(NDim,PopSize);  % 详解: 赋值：将 ones(...) 的结果保存到 vmax

for i=1:NDim  % 详解: for 循环：迭代变量 i 遍历 1:NDim
    vmax(i,:)=(UpperBound(i,:)-LowerBound(i,:))/10;  % 详解: 执行语句
end  % 详解: 执行语句
velocity = vmax.*rand(1);  % 详解: 赋值：计算表达式并保存到 velocity


for i = 1:PopSize,  % 详解: for 循环：迭代变量 i 遍历 1:PopSize,
    fvalue(i) = fname(population(:,i));  % 详解: 调用函数：fvalue(i) = fname(population(:,i))
end  % 详解: 执行语句

pbest = population;  % 详解: 赋值：计算表达式并保存到 pbest
fpbest = fvalue;  % 详解: 赋值：计算表达式并保存到 fpbest
[fbestval,index] = min(fvalue);  % 详解: 统计：最大/最小值

while(iteration < MaxIter)  % 详解: 调用函数：while(iteration < MaxIter)
    iteration = iteration +1;  % 详解: 赋值：计算表达式并保存到 iteration
       

    R1 = rand(NDim, PopSize);  % 详解: 赋值：将 rand(...) 的结果保存到 R1
    R2 = rand(NDim, PopSize);  % 详解: 赋值：将 rand(...) 的结果保存到 R2

    
    for i = 1:PopSize,  % 详解: for 循环：迭代变量 i 遍历 1:PopSize,
        fvalue(i) = fname(population(:,i));  % 详解: 调用函数：fvalue(i) = fname(population(:,i))
    end  % 详解: 执行语句

    changeColumns = fvalue < fpbest;  % 详解: 赋值：计算表达式并保存到 changeColumns
    pbest(:, find(changeColumns)) = population(:, find(changeColumns));  % 详解: 调用函数：pbest(:, find(changeColumns)) = population(:, find(changeColumns))
    fpbest = fpbest.*( ~changeColumns) + fvalue.*changeColumns;  % 详解: 赋值：计算表达式并保存到 fpbest
        
    [fbestval, index] = min(fpbest);  % 详解: 统计：最大/最小值
    

    for i=1:PopSize  % 详解: for 循环：迭代变量 i 遍历 1:PopSize
        gbest(:,i) = population(:,index);  % 详解: 调用函数：gbest(:,i) = population(:,index)
    end  % 详解: 执行语句

    velocity = w*velocity + c1*R1.*(pbest-population) + c2*R2.*(gbest-population);  % 详解: 赋值：计算表达式并保存到 velocity
    
    velocity=(velocity<-vmax).*(-vmax)+(velocity>vmax).*(vmax)+(velocity>-vmax & velocity<vmax).*velocity;  % 详解: 赋值：计算表达式并保存到 velocity
    
    
    population = population + velocity;  % 详解: 赋值：计算表达式并保存到 population
    
    population(population>UpperBound)=UpperBound(population>UpperBound);  % 详解: 调用函数：population(population>UpperBound)=UpperBound(population>UpperBound)
    population(population<LowerBound)=LowerBound(population<LowerBound);  % 详解: 调用函数：population(population<LowerBound)=LowerBound(population<LowerBound)
    
   fprintf('%d\t%i\n',iteration,fbestval);  % 详解: 调用函数：fprintf('%d\t%i\n',iteration,fbestval)
   his(iteration) = fbestval;  % 详解: 执行语句
end  % 详解: 执行语句
   plot(his,'b');  % 详解: 调用函数：plot(his,'b')
   xlabel('Iteration');  % 详解: 调用函数：xlabel('Iteration')
   ylabel('Fvalue');  % 详解: 调用函数：ylabel('Fvalue')
   hold on;  % 详解: 执行语句
end  % 详解: 执行语句








