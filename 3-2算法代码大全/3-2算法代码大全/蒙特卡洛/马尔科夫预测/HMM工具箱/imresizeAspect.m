% 文件: imresizeAspect.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function img = imresizeAspect(img, maxSize)  % 详解: 执行语句


if isempty(maxSize), return; end  % 详解: 条件判断：if (isempty(maxSize), return; end)

[y x c] = size(img);  % 详解: 获取向量/矩阵尺寸
a= y/x;  % 详解: 赋值：计算表达式并保存到 a
yy = maxSize(1); xx = maxSize(2);  % 详解: 赋值：将 maxSize(...) 的结果保存到 yy
if y <= yy & x <= xx  % 详解: 条件判断：if (y <= yy & x <= xx)
else  % 详解: 条件判断：else 分支
  if a < 1  % 详解: 条件判断：if (a < 1)
    img = imresize(img, ceil([a*xx xx]), 'bilinear');  % 详解: 赋值：将 imresize(...) 的结果保存到 img
  else  % 详解: 条件判断：else 分支
    img = imresize(img, ceil([yy yy/a]), 'bilinear');  % 详解: 赋值：将 imresize(...) 的结果保存到 img
  end  % 详解: 执行语句
  fprintf('resizing from %dx%d to %dx%d\n', y, x, size(img,1), size(img,2));  % 详解: 调用函数：fprintf('resizing from %dx%d to %dx%d\n', y, x, size(img,1), size(img,2))
end  % 详解: 执行语句


if 0  % 详解: 条件判断：if (0)
  maxSize = [240 320];  % 详解: 赋值：计算表达式并保存到 maxSize
  img = imread('C:\Images\Wearables\Database_static_street\art11.jpg');  % 详解: 赋值：将 imread(...) 的结果保存到 img
  img2 = imresizeAspect(img, maxSize);  % 详解: 赋值：将 imresizeAspect(...) 的结果保存到 img2
  figure(1); clf; imshow(img)  % 详解: 调用函数：figure(1); clf; imshow(img)
  figure(2); clf; imshow(img2)  % 详解: 调用函数：figure(2); clf; imshow(img2)
  fprintf('%dx%d (%5.3f) to %dx%d (%5.3f)\n', ...  % 详解: 打印/显示输出
	  size(img,1), size(img,2), size(img,1)/size(img,2), ...  % 详解: 获取向量/矩阵尺寸
	  size(img2,1), size(img2,2), size(img2,1)/size(img2,2));  % 详解: 调用函数：size(img2,1), size(img2,2), size(img2,1)/size(img2,2))
end  % 详解: 执行语句




