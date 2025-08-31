% 文件: montageKPM.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function h = montageKPM(arg)  % 详解: 执行语句

if iscell(arg)  % 详解: 条件判断：if (iscell(arg))
  h= montageFilenames(arg);  % 详解: 赋值：将 montageFilenames(...) 的结果保存到 h
else  % 详解: 条件判断：else 分支
  nr = size(arg,1); nc = size(arg,2); Npatches = size(arg,3);  % 详解: 赋值：将 size(...) 的结果保存到 nr
  patchesColor = reshape(arg, [nr nc 1 Npatches]);  % 详解: 赋值：将 reshape(...) 的结果保存到 patchesColor
  patchesColor = patchesColor ./ max(patchesColor(:));  % 详解: 赋值：计算表达式并保存到 patchesColor
  
  if 1  % 详解: 条件判断：if (1)
    border = 5;  % 详解: 赋值：计算表达式并保存到 border
    bgColor = ones(1,1,class(patchesColor));  % 详解: 赋值：将 ones(...) 的结果保存到 bgColor
    patchesColorBig = bgColor*ones(nr+2*border, nc+2*border, 1, Npatches, class(patchesColor));  % 详解: 赋值：计算表达式并保存到 patchesColorBig
    patchesColorBig(border+1:end-border, border+1:end-border, :, :) = patchesColor;  % 详解: 执行语句
  else  % 详解: 条件判断：else 分支
    patchesColorBig = patchesColor;  % 详解: 赋值：计算表达式并保存到 patchesColorBig
  end  % 详解: 执行语句
  montage(patchesColorBig)  % 详解: 调用函数：montage(patchesColorBig)

end  % 详解: 执行语句


function h = montageFilenames(filenames)  % 详解: 执行语句


aspectRatio = 1;  % 详解: 赋值：计算表达式并保存到 aspectRatio
nMontageCols = sqrt(aspectRatio * nRows * nFrames / nCols);  % 详解: 赋值：将 sqrt(...) 的结果保存到 nMontageCols

nMontageCols = ceil(nMontageCols);  % 详解: 赋值：将 ceil(...) 的结果保存到 nMontageCols
nMontageRows = ceil(nFrames / nMontageCols);  % 详解: 赋值：将 ceil(...) 的结果保存到 nMontageRows

b = a(1,1);  % 详解: 赋值：将 a(...) 的结果保存到 b
b(1,1) = 0;  % 详解: 执行语句
b = repmat(b, [nMontageRows*nRows, nMontageCols*nCols, nBands, 1]);  % 详解: 赋值：将 repmat(...) 的结果保存到 b

rows = 1 : nRows;  % 详解: 赋值：计算表达式并保存到 rows
cols = 1 : nCols;  % 详解: 赋值：计算表达式并保存到 cols

for i = 0:nMontageRows-1  % 详解: for 循环：迭代变量 i 遍历 0:nMontageRows-1
  for j = 0:nMontageCols-1,  % 详解: for 循环：迭代变量 j 遍历 0:nMontageCols-1,
    k = j + i * nMontageCols + 1;  % 详解: 赋值：计算表达式并保存到 k
    if k <= nFrames  % 详解: 条件判断：if (k <= nFrames)
      b(rows + i * nRows, cols + j * nCols, :) = a(:,:,:,k);  % 详解: 调用函数：b(rows + i * nRows, cols + j * nCols, :) = a(:,:,:,k)
    else  % 详解: 条件判断：else 分支
      break;  % 详解: 跳出循环：break
    end  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句

if isempty(cm)  % 详解: 条件判断：if (isempty(cm))
  hh = imshow(b);  % 详解: 赋值：将 imshow(...) 的结果保存到 hh
else  % 详解: 条件判断：else 分支
  hh = imshow(b,cm);  % 详解: 赋值：将 imshow(...) 的结果保存到 hh
end  % 详解: 执行语句

if nargout > 0  % 详解: 条件判断：if (nargout > 0)
    h = hh;  % 详解: 赋值：计算表达式并保存到 h
end  % 详解: 执行语句


function [I,map] = parse_inputs(varargin)  % 详解: 函数定义：parse_inputs(varargin), 返回：I,map

map = [];  % 详解: 赋值：计算表达式并保存到 map

iptchecknargin(1,2,nargin,mfilename);  % 详解: 调用函数：iptchecknargin(1,2,nargin,mfilename)
iptcheckinput(varargin{1},{'uint8' 'double' 'uint16' 'logical' 'single' ...  % 详解: 执行语句
                    'int16'},{},mfilename, 'I, BW, or RGB',1);  % 详解: 执行语句
I = varargin{1};  % 详解: 赋值：计算表达式并保存到 I

if nargin==2  % 详解: 条件判断：if (nargin==2)
  if isa(I,'int16')  % 详解: 条件判断：if (isa(I,'int16'))
    eid = sprintf('Images:%s:invalidIndexedImage',mfilename);  % 详解: 赋值：将 sprintf(...) 的结果保存到 eid
    msg1 = 'An indexed image can be uint8, uint16, double, single, or ';  % 详解: 赋值：计算表达式并保存到 msg1
    msg2 = 'logical.';  % 详解: 赋值：计算表达式并保存到 msg2
    error(eid,'%s %s',msg1, msg2);  % 详解: 调用函数：error(eid,'%s %s',msg1, msg2)
  end  % 详解: 执行语句
  map = varargin{2};  % 详解: 赋值：计算表达式并保存到 map
  iptcheckinput(map,{'double'},{},mfilename,'MAP',1);  % 详解: 调用函数：iptcheckinput(map,{'double'},{},mfilename,'MAP',1)
  if ((size(map,1) == 1) && (prod(map) == numel(I)))  % 详解: 条件判断：if (((size(map,1) == 1) && (prod(map) == numel(I))))
    eid = sprintf('Images:%s:obsoleteSyntax',mfilename);  % 详解: 赋值：将 sprintf(...) 的结果保存到 eid
    msg1 = 'MONTAGE(D,[M N P]) is an obsolete syntax.';  % 详解: 赋值：计算表达式并保存到 msg1
    msg2 = 'Use multidimensional arrays to represent multiframe images.';  % 详解: 赋值：计算表达式并保存到 msg2
    error(eid,'%s\n%s',msg1,msg2);  % 详解: 调用函数：error(eid,'%s\n%s',msg1,msg2)
  end  % 详解: 执行语句
end  % 详解: 执行语句




