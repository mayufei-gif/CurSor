% 文件: montageKPM3.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function montageKPM3(data)  % 详解: 函数定义：montageKPM3(data)

Nframes = length(data);  % 详解: 赋值：将 length(...) 的结果保存到 Nframes
Nbands = -inf;  % 详解: 赋值：计算表达式并保存到 Nbands
nr = -inf; nc = -inf;  % 详解: 赋值：计算表达式并保存到 nr
for f=1:Nframes  % 详解: for 循环：迭代变量 f 遍历 1:Nframes
  if isempty(data{f}), continue; end  % 详解: 条件判断：if (isempty(data{f}), continue; end)
  nr = max(nr, size(data{f},1));  % 详解: 赋值：将 max(...) 的结果保存到 nr
  nc = max(nc, size(data{f},2));  % 详解: 赋值：将 max(...) 的结果保存到 nc
  Nbands = max(Nbands, size(data{f},3));  % 详解: 赋值：将 max(...) 的结果保存到 Nbands
end  % 详解: 执行语句
data2 = zeros(nr, nc, Nbands, Nframes);  % 详解: 赋值：将 zeros(...) 的结果保存到 data2
for f=1:Nframes  % 详解: for 循环：迭代变量 f 遍历 1:Nframes
  if isempty(data{f}), continue; end  % 详解: 条件判断：if (isempty(data{f}), continue; end)
  data2(1:size(data{f},1), 1:size(data{f},2), :, f) = data{f};  % 详解: 获取向量/矩阵尺寸
end  % 详解: 执行语句

montageKPM2(data2)  % 详解: 调用函数：montageKPM2(data2)




