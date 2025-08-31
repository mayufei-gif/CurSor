% 文件: image_rgb.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function image_rgb(M)  % 详解: 函数定义：image_rgb(M)

cmap = [1 0 0;  % 详解: 赋值：计算表达式并保存到 cmap
	0 1 0;  % 详解: 执行语句
	0 0 1;  % 详解: 执行语句
	127/255 1 212/255];  % 详解: 执行语句
image(M)  % 详解: 调用函数：image(M)
set(gcf,'colormap', cmap);  % 详解: 调用函数：set(gcf,'colormap', cmap)

if 1  % 详解: 条件判断：if (1)
  str = {};  % 详解: 赋值：计算表达式并保存到 str
  for i=1:size(cmap,1)  % 详解: for 循环：迭代变量 i 遍历 1:size(cmap,1)
    dummy_handle(i) = line([0 0.1], [0 0.1]);  % 详解: 调用函数：dummy_handle(i) = line([0 0.1], [0 0.1])
    set(dummy_handle(i), 'color', cmap(i,:));  % 详解: 调用函数：set(dummy_handle(i), 'color', cmap(i,:))
    set(dummy_handle(i), 'linewidth', 2);  % 详解: 调用函数：set(dummy_handle(i), 'linewidth', 2)
    str{i} = num2str(i);  % 详解: 执行语句
  end  % 详解: 执行语句
  legend(dummy_handle, str, -1);  % 详解: 调用函数：legend(dummy_handle, str, -1)
end  % 详解: 执行语句

if 0  % 详解: 条件判断：if (0)
[nrows ncols] = size(M);  % 详解: 获取向量/矩阵尺寸
img = zeros(nrows, ncols, 3);  % 详解: 赋值：将 zeros(...) 的结果保存到 img
for r=1:nrows  % 详解: for 循环：迭代变量 r 遍历 1:nrows
  for c=1:ncols  % 详解: for 循环：迭代变量 c 遍历 1:ncols
    q = M(r,c);  % 详解: 赋值：将 M(...) 的结果保存到 q
    img(r,c,q) = 1;  % 详解: 执行语句
  end  % 详解: 执行语句
end  % 详解: 执行语句
image(img)  % 详解: 调用函数：image(img)
end  % 详解: 执行语句




