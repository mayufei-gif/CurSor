% 文件: TRIM.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function data=trim(data,outval)  % 详解: 执行语句


if nargin<2,outval=4;end  % 详解: 条件判断：if (nargin<2,outval=4;end)
outliers=(isnan(data)|abs(data)==inf);  % 详解: 赋值：计算表达式并保存到 outliers
[n,m]=size(data);mu=mean(data);sigma=std(data);  % 详解: 获取向量/矩阵尺寸
outliers=outliers+(abs(data-ones(n,1)*mu)>outval*ones(n,1)*sigma);  % 详解: 赋值：计算表达式并保存到 outliers
if m>1,  % 详解: 条件判断：if (m>1,)
   data(any(outliers'),:)=[];  % 调用函数：data  % 详解: 执行语句  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  data(find(outliers'),:)=[];  % 调用函数：data  % 详解: 执行语句  % 详解: 执行语句
end  % 详解: 执行语句





