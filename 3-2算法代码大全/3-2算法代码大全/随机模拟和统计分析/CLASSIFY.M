% 文件: CLASSIFY.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function class = classify(sample,training,group)  % 详解: 执行语句


[gr,gc] = size(group);  % 详解: 获取向量/矩阵尺寸
if min(gr,gc) ~= 1  % 详解: 条件判断：if (min(gr,gc) ~= 1)
   error('Requires a vector third argument.');  % 详解: 调用函数：error('Requires a vector third argument.')
end  % 详解: 执行语句

if gc ~= 1,  % 详解: 条件判断：if (gc ~= 1,)
   group = group(:);  % 详解: 赋值：将 group(...) 的结果保存到 group
   gr = gc;  % 详解: 赋值：计算表达式并保存到 gr
end  % 详解: 执行语句

if any(group - round(group)) | any(group) < 1  % 详解: 条件判断：if (any(group - round(group)) | any(group) < 1)
  error('The third input argument must be positive integers.');  % 详解: 调用函数：error('The third input argument must be positive integers.')
end  % 详解: 执行语句
maxg = max(group);  % 详解: 赋值：将 max(...) 的结果保存到 maxg

[tr,tc] = size(training);  % 详解: 获取向量/矩阵尺寸
if tr ~= gr,  % 详解: 条件判断：if (tr ~= gr,)
  error('The number of rows in the second and third input arguments must match.');  % 详解: 调用函数：error('The number of rows in the second and third input arguments must match.')
end  % 详解: 执行语句

[sr,sc] = size(sample);  % 详解: 获取向量/矩阵尺寸
if sc ~= tc  % 详解: 条件判断：if (sc ~= tc)
  error('The number of columns in the first and second input arguments must match.');  % 详解: 调用函数：error('The number of columns in the first and second input arguments must match.')
end  % 详解: 执行语句

d = zeros(sr,maxg);  % 详解: 赋值：将 zeros(...) 的结果保存到 d
for k = 1:maxg  % 详解: for 循环：迭代变量 k 遍历 1:maxg
   groupk = training(find(group == k),:);  % 详解: 赋值：将 training(...) 的结果保存到 groupk
   d(:,k) = mahal(sample,groupk);  % 详解: 调用函数：d(:,k) = mahal(sample,groupk)
end  % 详解: 执行语句
[tmp, class] = min(d');  % 统计：最大/最小值  % 详解: 统计：最大/最小值  % 详解: 统计：最大/最小值
class = class';  % 赋值：设置变量 class  % 详解: 赋值：计算表达式并保存到 class  % 详解: 赋值：计算表达式并保存到 class




