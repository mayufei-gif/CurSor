% 文件: netplot.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%函数名netplot
%使用方法输入请help netplot
%无返回值
%函数只能处理无向图
%作者：tiandsp
%最后修改：2012.12.26
function netplot(A,flag)  % 详解: 函数定义：netplot(A,flag)
    if flag==1  % 详解: 条件判断：if (flag==1)
        ND_netplot(A);  % 详解: 调用函数：ND_netplot(A)
        return;  % 详解: 返回：从当前函数返回
    end  % 详解: 执行语句
    
    if flag==2  % 详解: 条件判断：if (flag==2)
        [m n]=size(A);  % 详解: 获取向量/矩阵尺寸
        W=zeros(m,m);  % 详解: 赋值：将 zeros(...) 的结果保存到 W
        for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
            a=find(A(:,i)~=0);  % 详解: 赋值：将 find(...) 的结果保存到 a
            W(a(1),a(2))=1;  % 详解: 执行语句
            W(a(2),a(1))=1;  % 详解: 执行语句
        end  % 详解: 执行语句
        ND_netplot(W);  % 详解: 调用函数：ND_netplot(W)
        return;  % 详解: 返回：从当前函数返回
    end  % 详解: 执行语句
           
    function ND_netplot(A)  % 详解: 函数定义：ND_netplot(A)
        [n n]=size(A);  % 详解: 获取向量/矩阵尺寸
        w=floor(sqrt(n));  % 详解: 赋值：将 floor(...) 的结果保存到 w
        h=floor(n/w);  % 详解: 赋值：将 floor(...) 的结果保存到 h
        x=[];  % 详解: 赋值：计算表达式并保存到 x
        y=[];  % 详解: 赋值：计算表达式并保存到 y
        for i=1:h  % 详解: for 循环：迭代变量 i 遍历 1:h
            for j=1:w  % 详解: for 循环：迭代变量 j 遍历 1:w
                x=[x 10*rand(1)+(j-1)*10];  % 详解: 赋值：计算表达式并保存到 x
                y=[y 10*rand(1)+(i-1)*10];  % 详解: 赋值：计算表达式并保存到 y
            end  % 详解: 执行语句
        end  % 详解: 执行语句
        ed=n-h*w;  % 详解: 赋值：计算表达式并保存到 ed
        for i=1:ed  % 详解: for 循环：迭代变量 i 遍历 1:ed
           x=[x 10*rand(1)+(i-1)*10];  % 详解: 赋值：计算表达式并保存到 x
           y=[y 10*rand(1)+h*10];  % 详解: 赋值：计算表达式并保存到 y
        end  % 详解: 执行语句
        plot(x,y,'r*');  % 详解: 调用函数：plot(x,y,'r*')
        
        title('网络拓扑图');  % 详解: 调用函数：title('网络拓扑图')
        for i=1:n  % 详解: for 循环：迭代变量 i 遍历 1:n
            for j=i:n  % 详解: for 循环：迭代变量 j 遍历 i:n
                if A(i,j)~=0  % 详解: 条件判断：if (A(i,j)~=0)
                    c=num2str(A(i,j));  % 详解: 赋值：将 num2str(...) 的结果保存到 c
                    text((x(i)+x(j))/2,(y(i)+y(j))/2,c,'Fontsize',10);  % 详解: 调用函数：text((x(i)+x(j))/2,(y(i)+y(j))/2,c,'Fontsize',10)
                    line([x(i) x(j)],[y(i) y(j)]);  % 详解: 调用函数：line([x(i) x(j)],[y(i) y(j)])
                end  % 详解: 执行语句
                text(x(i),y(i),num2str(i),'Fontsize',14,'color','r');  % 详解: 调用函数：text(x(i),y(i),num2str(i),'Fontsize',14,'color','r')
                hold on;  % 详解: 执行语句
            end  % 详解: 执行语句
        end  % 详解: 执行语句
    end  % 详解: 执行语句
    
end  % 详解: 执行语句



