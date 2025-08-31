% 文件: BGf.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [f wf zwf]=BGf(C,b)  % 详解: 函数定义：BGf(C,b), 返回：f wf zwf


n=size(C,2);  % 详解: 赋值：将 size(...) 的结果保存到 n
wf=0;wf0=Inf;  % 详解: 赋值：计算表达式并保存到 wf
f=zeros(n,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 f
while(1)  % 详解: 调用函数：while(1)
      a=ones(n,n)*inf;  % 详解: 赋值：将 ones(...) 的结果保存到 a
      for(i=1:n)  % 详解: 调用函数：for(i=1:n)
           a(i,i)=0;  % 详解: 执行语句
      end  % 详解: 执行语句
      for(i=1:n)  % 详解: 调用函数：for(i=1:n)
          for(j=1:n)  % 详解: 调用函数：for(j=1:n)
              if(C(i,j)>0&f(i,j)==0)  % 详解: 调用函数：if(C(i,j)>0&f(i,j)==0)
                  a(i,j)=b(i,j);  % 详解: 调用函数：a(i,j)=b(i,j)
              elseif(C(i,j)>0&f(i,j)==C(i,j))  % 详解: 调用函数：elseif(C(i,j)>0&f(i,j)==C(i,j))
                  a(j,i)=-b(i,j);  % 详解: 调用函数：a(j,i)=-b(i,j)
              elseif(C(i,j)>0)a(i,j)=b(i,j);  % 详解: 调用函数：elseif(C(i,j)>0)a(i,j)=b(i,j)
                  a(j,i)=-b(i,j);  % 详解: 调用函数：a(j,i)=-b(i,j)
              end  % 详解: 执行语句
          end  % 详解: 执行语句
      end  % 详解: 执行语句
      for(i=2:n)  % 详解: 调用函数：for(i=2:n)
          p(i)=Inf;s(i)=i;  % 详解: 执行语句
      end  % 详解: 执行语句
      for(k=1:n)  % 详解: 调用函数：for(k=1:n)
          pd=1;  % 详解: 赋值：计算表达式并保存到 pd
          for(i=2:n)  % 详解: 调用函数：for(i=2:n)
              for(j=1:n)  % 详解: 调用函数：for(j=1:n)
                  if(p(i)>p(j)+a(j,i))  % 详解: 调用函数：if(p(i)>p(j)+a(j,i))
                      p(i)=p(j)+a(j,i);s(i)=j;pd=0;  % 详解: 执行语句
                  end  % 详解: 执行语句
              end  % 详解: 执行语句
          end  % 详解: 执行语句
          if(pd)  % 详解: 调用函数：if(pd)
              break;  % 详解: 跳出循环：break
          end  % 详解: 执行语句
      end  % 详解: 执行语句
      if(p(n)==Inf)  % 详解: 调用函数：if(p(n)==Inf)
          break;  % 详解: 跳出循环：break
      end  % 详解: 执行语句
    dvt=Inf;t=n;  % 详解: 赋值：计算表达式并保存到 dvt
while(1)  % 详解: 调用函数：while(1)
      if(a(s(t),t)>0)  % 详解: 调用函数：if(a(s(t),t)>0)
          dvtt=C(s(t),t)-f(s(t),t);  % 详解: 赋值：将 C(...) 的结果保存到 dvtt
      elseif(a(s(t),t)<0)  % 详解: 调用函数：elseif(a(s(t),t)<0)
          dvtt=f(t,s(t));  % 详解: 赋值：将 f(...) 的结果保存到 dvtt
      end  % 详解: 执行语句
      if(dvt>dvtt)  % 详解: 调用函数：if(dvt>dvtt)
          dvt=dvtt;  % 详解: 赋值：计算表达式并保存到 dvt
      end  % 详解: 执行语句
      if(s(t)==1)  % 详解: 调用函数：if(s(t)==1)
          break;  % 详解: 跳出循环：break
      end  % 详解: 执行语句
      t=s(t);  % 详解: 赋值：将 s(...) 的结果保存到 t
end  % 详解: 执行语句
pd=0;  % 详解: 赋值：计算表达式并保存到 pd
if(wf+dvt>=wf0)  % 详解: 调用函数：if(wf+dvt>=wf0)
    dvt=wf0-wf;pd=1;  % 详解: 赋值：计算表达式并保存到 dvt
end  % 详解: 执行语句
t=n;  % 详解: 赋值：计算表达式并保存到 t
while(1)  % 详解: 调用函数：while(1)
    if(a(s(t),t)>0)  % 详解: 调用函数：if(a(s(t),t)>0)
        f(s(t),t)=f(s(t),t)+dvt;  % 详解: 执行语句
    elseif(a(s(t),t)<0)  % 详解: 调用函数：elseif(a(s(t),t)<0)
        f(t,s(t))=f(t,s(t))-dvt;  % 详解: 执行语句
    end  % 详解: 执行语句
    if(s(t)==1)  % 详解: 调用函数：if(s(t)==1)
        break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
    t=s(t);end  % 详解: 赋值：将 s(...) 的结果保存到 t
    if(pd)  % 详解: 调用函数：if(pd)
        break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
    wf=0;  % 详解: 赋值：计算表达式并保存到 wf
    for(j=1:n)  % 详解: 调用函数：for(j=1:n)
        wf=wf+f(1,j);  % 详解: 赋值：计算表达式并保存到 wf
    end  % 详解: 执行语句
end  % 详解: 执行语句
zwf=0;  % 详解: 赋值：计算表达式并保存到 zwf
for(i=1:n)  % 详解: 调用函数：for(i=1:n)
    for(j=1:n)  % 详解: 调用函数：for(j=1:n)
        zwf=zwf+b(i,j)*f(i,j);  % 详解: 赋值：计算表达式并保存到 zwf
    end  % 详解: 执行语句
end  % 详解: 执行语句
f;  % 详解: 执行语句




