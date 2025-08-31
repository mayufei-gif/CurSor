% 文件: loadcell.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [lc,dflag,dattype]=loadcell(fname,delim,exclusions,options);  % 详解: 函数定义：loadcell(fname,delim,exclusions,options), 返回：lc,dflag,dattype

if (nargin<4)  % 详解: 条件判断：if ((nargin<4))
    options=' ';  % 详解: 赋值：计算表达式并保存到 options
end;  % 详解: 执行语句
dflag = [];  % 详解: 赋值：计算表达式并保存到 dflag
fid=fopen(fname,'rt');  % 详解: 赋值：将 fopen(...) 的结果保存到 fid
if (fid<0)  % 详解: 条件判断：if ((fid<0))
  lc=-1;  % 详解: 赋值：计算表达式并保存到 lc
else  % 详解: 条件判断：else 分支
  fullfile=fread(fid,'uchar=>char')';  % 赋值：设置变量 fullfile  % 详解: 赋值：将 fread(...) 的结果保存到 fullfile  % 详解: 赋值：将 fread(...) 的结果保存到 fullfile
  if ~isempty(findstr(options,'free'))  % 详解: 条件判断：if (~isempty(findstr(options,'free')))
      fullfile=strrep(fullfile,char(10),'');  % 详解: 赋值：将 strrep(...) 的结果保存到 fullfile
  end;  % 详解: 执行语句
  delimpos=[];  % 详解: 赋值：计算表达式并保存到 delimpos
  for s=1:length(delim)  % 详解: for 循环：迭代变量 s 遍历 1:length(delim)
    delimpos=[delimpos find(fullfile==delim(s))];  % 详解: 赋值：计算表达式并保存到 delimpos
  end  % 详解: 执行语句
  endpos=find(fullfile==char(10));  % 详解: 赋值：将 find(...) 的结果保存到 endpos
  endpos=setdiff(endpos,delimpos);  % 详解: 赋值：将 setdiff(...) 的结果保存到 endpos
  xclpos=[];  % 详解: 赋值：计算表达式并保存到 xclpos
  for s=1:length(exclusions);  % 详解: for 循环：迭代变量 s 遍历 1:length(exclusions);
    xclpos=[xclpos find(fullfile==exclusions(s))];  % 详解: 赋值：计算表达式并保存到 xclpos
  end  % 详解: 执行语句
  sort(xclpos);  % 详解: 调用函数：sort(xclpos)
  xclpos=[xclpos(1:2:end-1);xclpos(2:2:end)];  % 详解: 赋值：计算表达式并保存到 xclpos
  jointpos=union(delimpos,endpos);  % 详解: 赋值：将 union(...) 的结果保存到 jointpos
  t=1;  % 详解: 赋值：计算表达式并保存到 t
  removedelim=[];  % 详解: 赋值：计算表达式并保存到 removedelim
  for s=1:length(jointpos)  % 详解: for 循环：迭代变量 s 遍历 1:length(jointpos)
    if any((jointpos(s)>xclpos(1,:)) & (jointpos(s)<xclpos(2,:)))  % 详解: 条件判断：if (any((jointpos(s)>xclpos(1,:)) & (jointpos(s)<xclpos(2,:))))
      removedelim(t)=jointpos(s);  % 详解: 调用函数：removedelim(t)=jointpos(s)
      t=t+1;  % 详解: 赋值：计算表达式并保存到 t
    end;  % 详解: 执行语句

  end  % 详解: 执行语句
  jointpos=[0 setdiff(jointpos,removedelim)];  % 详解: 赋值：计算表达式并保存到 jointpos
  i=1;  % 详解: 赋值：计算表达式并保存到 i
  j=1;  % 详解: 赋值：计算表达式并保存到 j
  posind=1;  % 详解: 赋值：计算表达式并保存到 posind
  multflag=isempty(findstr(options,'single'));  % 详解: 赋值：将 isempty(...) 的结果保存到 multflag
  stringflag=~isempty(findstr(options,'string'));  % 详解: 赋值：计算表达式并保存到 stringflag
  emptnum=~isempty(findstr(options,'empty2num'));  % 详解: 赋值：计算表达式并保存到 emptnum
  while (posind<(length(jointpos)))  % 详解: while 循环：当 ((posind<(length(jointpos)))) 为真时迭代
    tempstr=fullfile(jointpos(posind)+1:jointpos(posind+1)-1);  % 详解: 赋值：将 fullfile(...) 的结果保存到 tempstr
    if ~(isempty(tempstr) & multflag);  % 详解: 条件判断：if (~(isempty(tempstr) & multflag);)
      dflag(i,j)=1;  % 详解: 执行语句
      tempno=str2double([tempstr]);  % 详解: 赋值：将 str2double(...) 的结果保存到 tempno
      if (isempty(tempstr) & emptnum)  % 详解: 条件判断：if ((isempty(tempstr) & emptnum))
          tempno=0;  % 详解: 赋值：计算表达式并保存到 tempno
      end;  % 详解: 执行语句
      dattype(i,j)=tempno;  % 详解: 执行语句
      if (isnan(tempno) |  stringflag)  % 详解: 条件判断：if ((isnan(tempno) |  stringflag))
        lc{i,j}=tempstr;  % 详解: 执行语句
      else  % 详解: 条件判断：else 分支
        lc{i,j}=tempno;  % 详解: 执行语句
      end;  % 详解: 执行语句
      j=j+1;  % 详解: 赋值：计算表达式并保存到 j
    end;  % 详解: 执行语句
    if ismember(jointpos(posind+1),endpos)  % 详解: 条件判断：if (ismember(jointpos(posind+1),endpos))
        i=i+1;  % 详解: 赋值：计算表达式并保存到 i
        j=1;  % 详解: 赋值：计算表达式并保存到 j
    end;  % 详解: 执行语句
    posind=posind+1;  % 详解: 赋值：计算表达式并保存到 posind
  end;  % 详解: 执行语句
end;  % 详解: 执行语句
dflag=logical(dflag);  % 详解: 赋值：将 logical(...) 的结果保存到 dflag




