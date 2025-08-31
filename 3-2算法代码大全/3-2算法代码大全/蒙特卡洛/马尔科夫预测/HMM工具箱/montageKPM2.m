% 文件: montageKPM2.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function montageKPM2(data)  % 详解: 函数定义：montageKPM2(data)

if ndims(data)==3  % 详解: 条件判断：if (ndims(data)==3)
  nr = size(data,1); nc = size(data,2); Npatches = size(data,3);  % 详解: 赋值：将 size(...) 的结果保存到 nr
  data = reshape(data, [nr nc 1 Npatches]);  % 详解: 赋值：将 reshape(...) 的结果保存到 data
else  % 详解: 条件判断：else 分支
  nr = size(data,1); nc = size(data,2); nbands = size(data,3); Npatches = size(data,4);  % 详解: 赋值：将 size(...) 的结果保存到 nr
end  % 详解: 执行语句
nativeVal = data(1, 1);  % 详解: 赋值：将 data(...) 的结果保存到 nativeVal
dataOrig = data;  % 详解: 赋值：计算表达式并保存到 dataOrig

border = 5;  % 详解: 赋值：计算表达式并保存到 border
bgColor = min(data(:));  % 详解: 赋值：将 min(...) 的结果保存到 bgColor
data = bgColor*ones(nr+2*border, nc+2*border, 1, Npatches, class(data));  % 详解: 赋值：计算表达式并保存到 data
data(border+1:end-border, border+1:end-border, :, :) = dataOrig;  % 详解: 执行语句

[width, height, bands, nFrames] = size(data);  % 详解: 获取向量/矩阵尺寸

axCols = sqrt(nFrames);  % 详解: 赋值：将 sqrt(...) 的结果保存到 axCols
if (axCols<1)  % 详解: 条件判断：if ((axCols<1))
    axCols = 1;  % 详解: 赋值：计算表达式并保存到 axCols
end  % 详解: 执行语句
axRows = nFrames/axCols;  % 详解: 赋值：计算表达式并保存到 axRows
if (ceil(axCols)-axCols) < (ceil(axRows)-axRows),  % 详解: 条件判断：if ((ceil(axCols)-axCols) < (ceil(axRows)-axRows),)
    axCols = ceil(axCols);  % 详解: 赋值：将 ceil(...) 的结果保存到 axCols
    axRows = ceil(nFrames/axCols);  % 详解: 赋值：将 ceil(...) 的结果保存到 axRows
else  % 详解: 条件判断：else 分支
    axRows = ceil(axRows);  % 详解: 赋值：将 ceil(...) 的结果保存到 axRows
    axCols = ceil(nFrames/axRows);  % 详解: 赋值：将 ceil(...) 的结果保存到 axCols
end  % 详解: 执行语句

storage = repmat(nativeVal, [axRows*width, axCols*height, bands, 1]);  % 详解: 赋值：将 repmat(...) 的结果保存到 storage

rows = 1:width;  % 详解: 赋值：计算表达式并保存到 rows
cols = 1:height;  % 详解: 赋值：计算表达式并保存到 cols
for i=0:axRows-1,  % 详解: for 循环：迭代变量 i 遍历 0:axRows-1,
  for j=0:axCols-1,  % 详解: for 循环：迭代变量 j 遍历 0:axCols-1,
    k = j+i*axCols+1;  % 详解: 赋值：计算表达式并保存到 k
    if k<=nFrames,  % 详解: 条件判断：if (k<=nFrames,)
      storage(rows+i*width, cols+j*height, :) = data(:,:,:,k);  % 详解: 调用函数：storage(rows+i*width, cols+j*height, :) = data(:,:,:,k)
    else  % 详解: 条件判断：else 分支
      break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句


im = imagesc(storage);  % 详解: 赋值：将 imagesc(...) 的结果保存到 im

ax = get(im, 'Parent');  % 详解: 赋值：将 get(...) 的结果保存到 ax
fig = get(ax, 'Parent');  % 详解: 赋值：将 get(...) 的结果保存到 fig
set(ax, 'XTick', [], 'YTick', [])  % 详解: 调用函数：set(ax, 'XTick', [], 'YTick', [])
figure(fig)  % 详解: 调用函数：figure(fig)

if 0  % 详解: 条件判断：if (0)
    colormap(gray);  % 详解: 调用函数：colormap(gray)
end  % 详解: 执行语句




