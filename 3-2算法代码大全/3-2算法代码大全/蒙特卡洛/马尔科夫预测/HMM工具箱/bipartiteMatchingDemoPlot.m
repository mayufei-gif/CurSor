% 文件: bipartiteMatchingDemoPlot.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function bipartiteMatchingDemoPlot(sources, detections, a)  % 详解: 函数定义：bipartiteMatchingDemoPlot(sources, detections, a)

hold on  % 详解: 执行语句
p1 = size(sources,2);  % 详解: 赋值：将 size(...) 的结果保存到 p1
p2 = size(detections,2);  % 详解: 赋值：将 size(...) 的结果保存到 p2
for i=1:p1  % 详解: for 循环：迭代变量 i 遍历 1:p1
  h=text(sources(1,i), sources(2,i), sprintf('s%d', i));  % 详解: 赋值：将 text(...) 的结果保存到 h
  set(h, 'color', 'r');  % 详解: 调用函数：set(h, 'color', 'r')
end  % 详解: 执行语句
for i=1:p2  % 详解: for 循环：迭代变量 i 遍历 1:p2
  h=text(detections(1,i), detections(2,i), sprintf('d%d', i));  % 详解: 赋值：将 text(...) 的结果保存到 h
  set(h, 'color', 'b');  % 详解: 调用函数：set(h, 'color', 'b')
end  % 详解: 执行语句

if nargin < 3, return; end  % 详解: 条件判断：if (nargin < 3, return; end)

for i=1:p1  % 详解: for 循环：迭代变量 i 遍历 1:p1
  j = a(i);  % 详解: 赋值：将 a(...) 的结果保存到 j
  if j==0  % 详解: 条件判断：if (j==0)
    continue  % 详解: 继续下一次循环：continue
  end  % 详解: 执行语句
  line([sources(1,i) detections(1,j)], [sources(2,i) detections(2,j)])  % 详解: 调用函数：line([sources(1,i) detections(1,j)], [sources(2,i) detections(2,j)])
end  % 详解: 执行语句
axis_pct;  % 详解: 执行语句




