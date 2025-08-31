% 文件: TUTDEMO.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

echo off  % 详解: 执行语句


echo on; clc  % 详解: 执行语句

pause  % 详解: 执行语句
clc  % 详解: 执行语句
pause  % 详解: 执行语句


fun = 'exp(x(1)) * (4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1)';  % 详解: 赋值：计算表达式并保存到 fun

pause  % 详解: 执行语句


x0 = [-1 1];  % 详解: 赋值：计算表达式并保存到 x0

pause  % 详解: 执行语句


options = [];  % 详解: 赋值：计算表达式并保存到 options

pause  % 详解: 执行语句

[x, options] = fminu(fun, x0, options);  % 详解: 执行语句

x  % 详解: 执行语句
pause  % 详解: 执行语句

options(8)  % 详解: 调用函数：options(8)
pause  % 详解: 执行语句

options(10)  % 详解: 调用函数：options(10)

pause  % 详解: 执行语句
clc  % 详解: 执行语句
pause  % 详解: 执行语句


funf = 'f = exp(x(1)) * (4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);';  % 详解: 赋值：计算表达式并保存到 funf
fung = 'g = [1.5 + x(1)*x(2) - x(1) - x(2); -x(1)*x(2) - 10];';  % 详解: 赋值：计算表达式并保存到 fung
fun = [funf fung];  % 详解: 赋值：计算表达式并保存到 fun

pause  % 详解: 执行语句


x0 = [-1 1];  % 详解: 赋值：计算表达式并保存到 x0

pause  % 详解: 执行语句


options = [];  % 详解: 赋值：计算表达式并保存到 options

pause  % 详解: 执行语句

[x, options] = constr(fun, x0, options);  % 详解: 执行语句

x  % 详解: 执行语句
pause  % 详解: 执行语句

options(8)  % 详解: 调用函数：options(8)
pause  % 详解: 执行语句

g = [1.5 + x(1)*x(2) - x(1) - x(2); - x(1)*x(2) - 10]  % 详解: 赋值：计算表达式并保存到 g
pause  % 详解: 执行语句

options(10)  % 详解: 调用函数：options(10)
pause  % 详解: 执行语句
clc  % 详解: 执行语句
pause  % 详解: 执行语句


funf = 'f = exp(x(1)) * (4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);';  % 详解: 赋值：计算表达式并保存到 funf
fung = 'g = [1.5 + x(1)*x(2) - x(1) - x(2); -x(1)*x(2) - 10];';  % 详解: 赋值：计算表达式并保存到 fung
fun = [funf fung];  % 详解: 赋值：计算表达式并保存到 fun

pause  % 详解: 执行语句

  
x0 = [-1 1];  % 详解: 赋值：计算表达式并保存到 x0

pause  % 详解: 执行语句


options = [];  % 详解: 赋值：计算表达式并保存到 options

pause  % 详解: 执行语句


vlb = zeros(1,2);  % 详解: 赋值：将 zeros(...) 的结果保存到 vlb
vub = [];  % 详解: 赋值：计算表达式并保存到 vub

pause  % 详解: 执行语句

[x, options] = constr(fun, x0, options, vlb, vub);  % 详解: 执行语句

x  % 详解: 执行语句
pause  % 详解: 执行语句

options(8)  % 详解: 调用函数：options(8)
pause  % 详解: 执行语句

g = [1.5 + x(1)*x(2) - x(1) - x(2); - x(1)*x(2) - 10]  % 详解: 赋值：计算表达式并保存到 g
pause  % 详解: 执行语句

options(10)  % 详解: 调用函数：options(10)
pause  % 详解: 执行语句
clc  % 详解: 执行语句
pause  % 详解: 执行语句


funf = 'f = exp(x(1)) * (4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);';  % 详解: 赋值：计算表达式并保存到 funf
fung = 'g = [1.5 + x(1)*x(2) - x(1) - x(2); -x(1)*x(2) - 10];';  % 详解: 赋值：计算表达式并保存到 fung
fun = [funf fung];  % 详解: 赋值：计算表达式并保存到 fun

pause  % 详解: 执行语句

 
x0 = [-1 1];  % 详解: 赋值：计算表达式并保存到 x0

pause  % 详解: 执行语句


options = [];  % 详解: 赋值：计算表达式并保存到 options

pause  % 详解: 执行语句


dfdx1 = 'exp(x(1))*(4*x(1)^2+2*x(2)^2+4*x(1)*x(2)+2*x(2)+1)+4*exp(x(1))*(2*x(1)+x(2));';  % 详解: 赋值：计算表达式并保存到 dfdx1
dfdx2 = '4*exp(x(1))*(x(1)+x(2)+0.5);';  % 详解: 赋值：计算表达式并保存到 dfdx2
dfdg = '[x(2)-1, -x(2); x(1)-1, -x(1)];';  % 详解: 赋值：计算表达式并保存到 dfdg
grad = ['df=[',dfdx1, dfdx2,']; dg=', dfdg];  % 详解: 赋值：计算表达式并保存到 grad

pause  % 详解: 执行语句

[x, options] = constr(fun, x0, options, [], [], grad);  % 详解: 执行语句

x  % 详解: 执行语句
pause  % 详解: 执行语句

options(8)  % 详解: 调用函数：options(8)
pause  % 详解: 执行语句

g = [1.5 + x(1)*x(2) - x(1) - x(2); - x(1)*x(2) - 10]  % 详解: 赋值：计算表达式并保存到 g
pause  % 详解: 执行语句

options(10)  % 详解: 调用函数：options(10)

pause  % 详解: 执行语句
clc  % 详解: 执行语句
pause  % 详解: 执行语句


funf = 'f = exp(x(1)) * (4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1);';  % 详解: 赋值：计算表达式并保存到 funf
fung = 'g = [x(1) + x(2); 1.5 + x(1)*x(2) - x(1) - x(2); -x(1)*x(2) - 10];';  % 详解: 赋值：计算表达式并保存到 fung
fun = [funf fung];  % 详解: 赋值：计算表达式并保存到 fun

pause  % 详解: 执行语句


x0 = [-1 1];  % 详解: 赋值：计算表达式并保存到 x0

pause  % 详解: 执行语句



clear options  % 详解: 执行语句
options(13) = 1;  % 详解: 执行语句

pause  % 详解: 执行语句

[x, options] = constr(fun, x0, options);  % 详解: 执行语句

x  % 详解: 执行语句
pause  % 详解: 执行语句

options(8)  % 详解: 调用函数：options(8)
pause  % 详解: 执行语句

g = [x(1) + x(2); 1.5 + x(1)*x(2) - x(1) - x(2); -x(1)*x(2) - 10]  % 详解: 赋值：计算表达式并保存到 g
pause  % 详解: 执行语句

options(10)  % 详解: 调用函数：options(10)

pause  % 详解: 执行语句
clc  % 详解: 执行语句
pause  % 详解: 执行语句


fun = 'exp(x(1)) * (4*x(1)^2 + 2*x(2)^2 + 4*x(1)*x(2) + 2*x(2) + 1)';  % 详解: 赋值：计算表达式并保存到 fun

pause  % 详解: 执行语句


x0 = [-1 1];  % 详解: 赋值：计算表达式并保存到 x0

pause  % 详解: 执行语句

clear options  % 详解: 执行语句
options(2) = 1e-8;  % 详解: 执行语句
options(3) = 1e-8;  % 详解: 执行语句

pause  % 详解: 执行语句

[x, options] = fminu(fun, x0, options);  % 详解: 执行语句

x  % 详解: 执行语句
pause  % 详解: 执行语句

options(8)  % 详解: 调用函数：options(8)
pause  % 详解: 执行语句

options(10)  % 详解: 调用函数：options(10)

pause  % 详解: 执行语句


options = 1;  % 详解: 赋值：计算表达式并保存到 options

pause  % 详解: 执行语句

[x, options] = fminu(fun, x0, options);  % 详解: 执行语句

x  % 详解: 执行语句
pause  % 详解: 执行语句

options(8)  % 详解: 调用函数：options(8)
pause  % 详解: 执行语句

options(10)  % 详解: 调用函数：options(10)

pause  % 详解: 执行语句
echo off  % 详解: 执行语句
disp('End of demo')  % 详解: 调用函数：disp('End of demo')




