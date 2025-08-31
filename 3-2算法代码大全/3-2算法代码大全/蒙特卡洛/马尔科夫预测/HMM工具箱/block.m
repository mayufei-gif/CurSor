% 文件: block.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function sub = block(blocks, block_sizes)  % 详解: 执行语句

blocks = blocks(:)';  % 赋值：设置变量 blocks  % 详解: 赋值：将 blocks(...) 的结果保存到 blocks  % 详解: 赋值：将 blocks(...) 的结果保存到 blocks
block_sizes = block_sizes(:)';  % 赋值：设置变量 block_sizes  % 详解: 赋值：将 block_sizes(...) 的结果保存到 block_sizes  % 详解: 赋值：将 block_sizes(...) 的结果保存到 block_sizes
skip = [0 cumsum(block_sizes)];  % 详解: 赋值：计算表达式并保存到 skip
start = skip(blocks)+1;  % 详解: 赋值：将 skip(...) 的结果保存到 start
fin = start + block_sizes(blocks) - 1;  % 详解: 赋值：计算表达式并保存到 fin
sub = [];  % 详解: 赋值：计算表达式并保存到 sub
for j=1:length(blocks)  % 详解: for 循环：迭代变量 j 遍历 1:length(blocks)
  sub = [sub start(j):fin(j)];  % 详解: 赋值：计算表达式并保存到 sub
end  % 详解: 执行语句




