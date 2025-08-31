% 文件: SystemMatrix.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [W_ind,W_dat] = SystemMatrix(theta,N,P_num,delta)  % 详解: 函数定义：SystemMatrix(theta,N,P_num,delta), 返回：W_ind,W_dat

N2 = N^2;  % 详解: 赋值：计算表达式并保存到 N2
M = length(theta) * P_num;  % 详解: 赋值：将 length(...) 的结果保存到 M
W_ind = zeros(M,2 * N);  % 详解: 赋值：将 zeros(...) 的结果保存到 W_ind
W_dat = zeros(M,2 * N);  % 详解: 赋值：将 zeros(...) 的结果保存到 W_dat

t = (-(P_num-1)/2:(P_num-1)/2)*delta;  % 详解: 赋值：计算表达式并保存到 t
if N <= 10 && length(theta) <= 5  % 详解: 条件判断：if (N <= 10 && length(theta) <= 5)
    x = (-N/2: 1 :N/2)*delta;  % 详解: 赋值：计算表达式并保存到 x
    y = (-N/2: 1 :N/2)*delta;  % 详解: 赋值：计算表达式并保存到 y
    plot(x,meshgrid(y,x),'k');  % 详解: 调用函数：plot(x,meshgrid(y,x),'k')
    hold on;  % 详解: 执行语句
    plot(meshgrid(x,y),y,'k');  % 详解: 调用函数：plot(meshgrid(x,y),y,'k')
    axis([-N/2-5,N/2+5,-N/2-5,N/2+5]);  % 详解: 调用函数：axis([-N/2-5,N/2+5,-N/2-5,N/2+5])
    text(0,-0.4*delta,'0');  % 详解: 调用函数：text(0,-0.4*delta,'0')
end  % 详解: 执行语句
for jj = 1: length(theta)  % 详解: for 循环：迭代变量 jj 遍历 1: length(theta)
    for ii = 1:1:P_num  % 详解: for 循环：迭代变量 ii 遍历 1:1:P_num
        
        u = zeros(1,2*N);v = zeros(1,2*N);  % 详解: 赋值：将 zeros(...) 的结果保存到 u
        th = theta(jj);  % 详解: 赋值：将 theta(...) 的结果保存到 th
        if th >=180|| th < 0  % 详解: 条件判断：if (th >=180|| th < 0)
            error('输入角度必须在0-180之间');  % 详解: 调用函数：error('输入角度必须在0-180之间')
        elseif th == 90  % 详解: 条件判断：elseif (th == 90)
                
                if N <=10 && length(theta) <=5  % 详解: 条件判断：if (N <=10 && length(theta) <=5)
                   xx = (-N/2-2:0.01:N/2+2)*delta;  % 详解: 赋值：计算表达式并保存到 xx
                   yy = t(ii);  % 详解: 赋值：将 t(...) 的结果保存到 yy
                   plot(xx,yy,'b');  % 详解: 调用函数：plot(xx,yy,'b')
                   hold on;  % 详解: 执行语句
                end  % 详解: 执行语句
                if t(ii) >= N/2 * delta || t(ii) <= -N/2 * delta;  % 详解: 条件判断：if (t(ii) >= N/2 * delta || t(ii) <= -N/2 * delta;)
                    continue;  % 详解: 继续下一次循环：continue
                end  % 详解: 执行语句
                kout = N * ceil(N/2-t(ii)/delta);  % 详解: 赋值：计算表达式并保存到 kout
                kk = (kout-(N-1)):1:kout;  % 详解: 赋值：计算表达式并保存到 kk
                u(1:N) = kk;  % 详解: 执行语句
                v(1:N) = ones(1,N) * delta;  % 详解: 创建全 1 矩阵/数组
        elseif th == 0  % 详解: 条件判断：elseif (th == 0)
            
            if N <= 10 && length(theta) <= 5  % 详解: 条件判断：if (N <= 10 && length(theta) <= 5)
                yy = (-N/2-2:0.01:N/2+2) * delta;  % 详解: 赋值：计算表达式并保存到 yy
                xx = t(ii);  % 详解: 赋值：将 t(...) 的结果保存到 xx
                plot(xx,yy,'b');  % 详解: 调用函数：plot(xx,yy,'b')
                hold on;  % 详解: 执行语句
            end  % 详解: 执行语句
            
            if t(ii) >= N/2 * delta || t(ii) <= -N/2 * delta;  % 详解: 条件判断：if (t(ii) >= N/2 * delta || t(ii) <= -N/2 * delta;)
                continue;  % 详解: 继续下一次循环：continue
            end  % 详解: 执行语句
            kin = ceil(N/2+t(ii)/delta);  % 详解: 赋值：将 ceil(...) 的结果保存到 kin
            kk = kin: N: (kin + N * (N-1));  % 详解: 赋值：计算表达式并保存到 kk
            u(1:N) = kk;  % 详解: 执行语句
            v(1:N) = ones(1,N) * delta;  % 详解: 创建全 1 矩阵/数组
        else  % 详解: 条件判断：else 分支
            if th > 90  % 详解: 条件判断：if (th > 90)
                th_temp = th - 90;  % 详解: 赋值：计算表达式并保存到 th_temp
            elseif th < 90  % 详解: 条件判断：elseif (th < 90)
                th_temp = 90 - th;  % 详解: 赋值：计算表达式并保存到 th_temp
            end  % 详解: 执行语句
            th_temp = th_temp * pi/180;  % 详解: 赋值：计算表达式并保存到 th_temp
            
            b = t/cos(th_temp);  % 详解: 赋值：计算表达式并保存到 b
            m = tan(th_temp);  % 详解: 赋值：将 tan(...) 的结果保存到 m
            y1d = -N/2 * delta * m + b(ii);  % 详解: 赋值：计算表达式并保存到 y1d
            y2d = N/2 * delta * m + b(ii);  % 详解: 赋值：计算表达式并保存到 y2d
            
            if N <= 10 && length(theta) <= 5  % 详解: 条件判断：if (N <= 10 && length(theta) <= 5)
                xx = (-N/2-2:0.01:N/2+2) * delta;  % 详解: 赋值：计算表达式并保存到 xx
                if th < 90  % 详解: 条件判断：if (th < 90)
                    yy = -m * xx + b(ii);  % 详解: 赋值：计算表达式并保存到 yy
                elseif th > 90  % 详解: 条件判断：elseif (th > 90)
                    yy = m * xx + b(ii);  % 详解: 赋值：计算表达式并保存到 yy
                end  % 详解: 执行语句
                plot(xx,yy,'b');  % 详解: 调用函数：plot(xx,yy,'b')
                hold on;  % 详解: 执行语句
            end  % 详解: 执行语句
            
            if(y1d < -N/2 * delta && y2d < -N/2 * delta) || (y1d > N/2 * delta && y2d > -N/2 * delta)  % 详解: 调用函数：if(y1d < -N/2 * delta && y2d < -N/2 * delta) || (y1d > N/2 * delta && y2d > -N/2 * delta)
                continue;  % 详解: 继续下一次循环：continue
            end  % 详解: 执行语句
        if y1d <= N/2 * delta && y1d >= -N/2 * delta && y2d > N/2 * delta  % 详解: 条件判断：if (y1d <= N/2 * delta && y1d >= -N/2 * delta && y2d > N/2 * delta)
            yin = y1d;  % 详解: 赋值：计算表达式并保存到 yin
            d1 = yin - floor(yin/delta) * delta;  % 详解: 赋值：计算表达式并保存到 d1
            kin = N * floor(N/2 - yin/delta) + 1;  % 详解: 赋值：计算表达式并保存到 kin
            yout = N/2 * delta;  % 详解: 赋值：计算表达式并保存到 yout
            xout = (yout - b(ii))/m;  % 详解: 赋值：计算表达式并保存到 xout
            kout = ceil(xout/delta) + N/2;  % 详解: 赋值：将 ceil(...) 的结果保存到 kout
        elseif y1d <= N/2 * delta && y1d >= -N/2 * delta && y2d >= -N/2 * delta && y2d < N/2 * delta  % 详解: 条件判断：elseif (y1d <= N/2 * delta && y1d >= -N/2 * delta && y2d >= -N/2 * delta && y2d < N/2 * delta)
            yin = y1d;  % 详解: 赋值：计算表达式并保存到 yin
            d1 = yin - floor(yin/delta) * delta;  % 详解: 赋值：计算表达式并保存到 d1
            kin = N * floor(N/2 - yin/delta) + 1;  % 详解: 赋值：计算表达式并保存到 kin
            yout = y2d;  % 详解: 赋值：计算表达式并保存到 yout
            kout = N * floor(N/2 - yout/delta) + N;  % 详解: 赋值：计算表达式并保存到 kout
        elseif y1d <- N/2 * delta && y2d >= N/2 * delta  % 详解: 条件判断：elseif (y1d <- N/2 * delta && y2d >= N/2 * delta)
            yin = -N/2 * delta;  % 详解: 赋值：计算表达式并保存到 yin
            xin = (yin - b(ii))/m;  % 详解: 赋值：计算表达式并保存到 xin
            d1 = N/2 * delta + (floor(xin/delta)*delta*m+b(ii));  % 详解: 赋值：计算表达式并保存到 d1
            kin = N * (N - 1) + N/2 + ceil(xin/delta);  % 详解: 赋值：计算表达式并保存到 kin
            yout = N/2 * delta;  % 详解: 赋值：计算表达式并保存到 yout
            xout = (yout - b(ii))/m;  % 详解: 赋值：计算表达式并保存到 xout
            kout = ceil(xout/delta) + N/2;  % 详解: 赋值：将 ceil(...) 的结果保存到 kout
        elseif y1d < -N/2 * delta && y2d >= -N/2 * delta && y2d < N/2 * delta  % 详解: 条件判断：elseif (y1d < -N/2 * delta && y2d >= -N/2 * delta && y2d < N/2 * delta)
            yin = -N/2 * delta;  % 详解: 赋值：计算表达式并保存到 yin
            xin = (yin - b(ii))/m;  % 详解: 赋值：计算表达式并保存到 xin
            d1 = N/2 * delta + (floor(xin/delta) * delta * m + b(ii));  % 详解: 赋值：计算表达式并保存到 d1
            kin = N * (N - 1) + N/2 + ceil(xin/delta);  % 详解: 赋值：计算表达式并保存到 kin
            yout = y2d;  % 详解: 赋值：计算表达式并保存到 yout
            kout = N * floor(N/2 - yout/delta) + N;  % 详解: 赋值：计算表达式并保存到 kout
        else  % 详解: 条件判断：else 分支
            continue  % 详解: 继续下一次循环：continue
        end  % 详解: 执行语句
      k = kin;  % 详解: 赋值：计算表达式并保存到 k
      c = 0;  % 详解: 赋值：计算表达式并保存到 c
      d2 = d1 + m * delta;  % 详解: 赋值：计算表达式并保存到 d2
      while k >=1 && k <= N2  % 详解: while 循环：当 (k >=1 && k <= N2) 为真时迭代
          c =c +1;  % 详解: 赋值：计算表达式并保存到 c
          if d1 >= 0 && d2 > delta  % 详解: 条件判断：if (d1 >= 0 && d2 > delta)
              u(c) = k;  % 详解: 执行语句
              v(c) = (delta - d1) * sqrt(m^2 + 1)/m;  % 详解: 执行语句
              if k > N && k ~= kout  % 详解: 条件判断：if (k > N && k ~= kout)
                  k = k - N;  % 详解: 赋值：计算表达式并保存到 k
                  d1 = d1 - delta;  % 详解: 赋值：计算表达式并保存到 d1
                  d2 = d1 + m * delta;  % 详解: 赋值：计算表达式并保存到 d2
              else  % 详解: 条件判断：else 分支
                  break;  % 详解: 跳出循环：break
              end  % 详解: 执行语句
          elseif d1 >= 0 && d2 == delta  % 详解: 条件判断：elseif (d1 >= 0 && d2 == delta)
              u(c) = k;  % 详解: 执行语句
              v(c) = delta * sqrt(m^2 + 1);  % 详解: 调用函数：v(c) = delta * sqrt(m^2 + 1)
              if k > N && k~= kout  % 详解: 条件判断：if (k > N && k~= kout)
                  k = k - N + 1;  % 详解: 赋值：计算表达式并保存到 k
                  d1 = 0;  % 详解: 赋值：计算表达式并保存到 d1
                  d2 = d1 + m * delta;  % 详解: 赋值：计算表达式并保存到 d2
              else  % 详解: 条件判断：else 分支
                  break;  % 详解: 跳出循环：break
              end  % 详解: 执行语句
          elseif d1 >= 0 && d2 < delta  % 详解: 条件判断：elseif (d1 >= 0 && d2 < delta)
              u(c) = k;  % 详解: 执行语句
              v(c) = delta * sqrt(m^2 + 1);  % 详解: 调用函数：v(c) = delta * sqrt(m^2 + 1)
              if k ~= kout  % 详解: 条件判断：if (k ~= kout)
                  k = k + 1;  % 详解: 赋值：计算表达式并保存到 k
                  d1 = d2;  % 详解: 赋值：计算表达式并保存到 d1
                  d2 = d1 + m * delta;  % 详解: 赋值：计算表达式并保存到 d2
              else  % 详解: 条件判断：else 分支
                  break;  % 详解: 跳出循环：break
              end  % 详解: 执行语句
          elseif d1 <= 0 && d2 >=0 && d2 <= delta  % 详解: 条件判断：elseif (d1 <= 0 && d2 >=0 && d2 <= delta)
              u(c) = k;  % 详解: 执行语句
              v(c) = delta * sqrt(m^2 + 1);  % 详解: 调用函数：v(c) = delta * sqrt(m^2 + 1)
              if k ~= kout  % 详解: 条件判断：if (k ~= kout)
                  k = k + 1;  % 详解: 赋值：计算表达式并保存到 k
                  d1 = d2;  % 详解: 赋值：计算表达式并保存到 d1
                  d2 = d1 + m * delta;  % 详解: 赋值：计算表达式并保存到 d2
              else  % 详解: 条件判断：else 分支
                  break;  % 详解: 跳出循环：break
              end  % 详解: 执行语句
          elseif d1 <= 0 && d2 > delta  % 详解: 条件判断：elseif (d1 <= 0 && d2 > delta)
              u(c) = k;  % 详解: 执行语句
              v(c) = delta * sqrt(m^2 + 1)/m;  % 详解: 执行语句
              if k > N && k ~= kout  % 详解: 条件判断：if (k > N && k ~= kout)
                  k = k - N;  % 详解: 赋值：计算表达式并保存到 k
                  d1 = -delta + d1;  % 详解: 赋值：计算表达式并保存到 d1
                  d2 = d1 + m * delta;  % 详解: 赋值：计算表达式并保存到 d2
              else  % 详解: 条件判断：else 分支
                  break;  % 详解: 跳出循环：break
              end  % 详解: 执行语句
          end  % 详解: 执行语句
      end  % 详解: 执行语句
      if th < 90  % 详解: 条件判断：if (th < 90)
          u_temp = zeros(1,2 * N);  % 详解: 赋值：将 zeros(...) 的结果保存到 u_temp
          if any(u) == 0  % 详解: 条件判断：if (any(u) == 0)
              continue;  % 详解: 继续下一次循环：continue
          end  % 详解: 执行语句
          ind = u >0;  % 详解: 赋值：计算表达式并保存到 ind
          for k = 1: length(u(ind))  % 详解: for 循环：迭代变量 k 遍历 1: length(u(ind))
              r = rem(u(k),N);  % 详解: 赋值：将 rem(...) 的结果保存到 r
              if r == 0  % 详解: 条件判断：if (r == 0)
                  u_temp(k) = u(k) - N + 1;  % 详解: 执行语句
              else  % 详解: 条件判断：else 分支
                  u_temp(k) = u(k) - 2 * r + N + 1;  % 详解: 执行语句
              end  % 详解: 执行语句
          end  % 详解: 执行语句
          u = u_temp;  % 详解: 赋值：计算表达式并保存到 u
      end  % 详解: 执行语句
        end  % 详解: 执行语句
        W_ind((jj-1)* P_num +ii,:) = u;  % 详解: 执行语句
        W_dat((jj-1)* P_num +ii,:) = v;  % 详解: 执行语句
    end  % 详解: 执行语句
end  % 详解: 执行语句
                
                



