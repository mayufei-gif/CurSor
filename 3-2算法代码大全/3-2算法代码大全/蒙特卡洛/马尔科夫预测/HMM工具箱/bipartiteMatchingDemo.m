% 文件: bipartiteMatchingDemo.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

% Consider matching sources to detections

%  s1 d2  
%         s2 d3
%  d1

%a  = bipartiteMatchingHungarian([52;0.01])

% sources(:,i) = [x y] coords
sources = [0.1 0.7; 0.6 0.4]';  % 赋值：设置变量 sources  % 详解: 赋值：计算表达式并保存到 sources  % 详解: 赋值：计算表达式并保存到 sources
detections = [0.2 0.2; 0.2 0.8; 0.7 0.1]';  % 赋值：设置变量 detections  % 详解: 赋值：计算表达式并保存到 detections  % 详解: 赋值：计算表达式并保存到 detections
dst = sqdist(sources, detections);  % 详解: 赋值：将 sqdist(...) 的结果保存到 dst

a = bipartiteMatchingHungarian(dst);  % 详解: 赋值：将 bipartiteMatchingHungarian(...) 的结果保存到 a
a2 = bipartiteMatchingIntProg(dst);  % 详解: 赋值：将 bipartiteMatchingIntProg(...) 的结果保存到 a2
assert(isequal(a(:),a2(:)))  % 详解: 调用函数：assert(isequal(a(:),a2(:)))


figure(1); clf  % 详解: 执行语句
bipartiteMatchingDemoPlot(sources, detections, a)  % 详解: 调用函数：bipartiteMatchingDemoPlot(sources, detections, a)





dst = sqdist(detections, sources);  % 详解: 赋值：将 sqdist(...) 的结果保存到 dst
a = bipartiteMatchingHungarian(dst);  % 详解: 赋值：将 bipartiteMatchingHungarian(...) 的结果保存到 a

a2 = bipartiteMatchingIntProg(dst);  % 详解: 赋值：将 bipartiteMatchingIntProg(...) 的结果保存到 a2
assert(isequal(a(:),a2(:)))  % 详解: 调用函数：assert(isequal(a(:),a2(:)))

figure(2); clf  % 详解: 执行语句
bipartiteMatchingDemoPlot(detections, sources, a)  % 详解: 调用函数：bipartiteMatchingDemoPlot(detections, sources, a)





sources = [0.1 0.3; 0.6 0.4]';  % 赋值：设置变量 sources  % 详解: 赋值：计算表达式并保存到 sources  % 详解: 赋值：计算表达式并保存到 sources
detections = [0.2 0.2; 0.2 0.8; 0.7 0.1]';  % 赋值：设置变量 detections  % 详解: 赋值：计算表达式并保存到 detections  % 详解: 赋值：计算表达式并保存到 detections
dst = sqdist(sources, detections);  % 详解: 赋值：将 sqdist(...) 的结果保存到 dst

a = bipartiteMatchingHungarian(dst);  % 详解: 赋值：将 bipartiteMatchingHungarian(...) 的结果保存到 a
[a2, ass] = bipartiteMatchingIntProg(dst);  % 详解: 执行语句
assert(isequal(a(:),a2(:)))  % 详解: 调用函数：assert(isequal(a(:),a2(:)))


figure(3); clf  % 详解: 执行语句
bipartiteMatchingDemoPlot(sources, detections, a)  % 详解: 调用函数：bipartiteMatchingDemoPlot(sources, detections, a)





randn('state', 0); rand('state', 0);  % 详解: 调用函数：randn('state', 0); rand('state', 0)
gmix = gmm(2, 2, 'spherical');  % 详解: 赋值：将 gmm(...) 的结果保存到 gmix
ndat1 = 10; ndat2 = 10; ndata = ndat1+ndat2;  % 详解: 赋值：计算表达式并保存到 ndat1
gmix.centres =  [0.5 0.5; 0.5 0.5];  % 详解: 赋值：计算表达式并保存到 gmix.centres
gmix.covars = [0.1 0.01];  % 详解: 赋值：计算表达式并保存到 gmix.covars
[x, label] = gmmsamp(gmix, ndata);  % 详解: 执行语句

ndx = find(label==1);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
sources = x(ndx,:)';  % 赋值：设置变量 sources  % 详解: 赋值：将 x(...) 的结果保存到 sources  % 详解: 赋值：将 x(...) 的结果保存到 sources
ndx = find(label==2);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
detections = x(ndx,:)';  % 赋值：设置变量 detections  % 详解: 赋值：将 x(...) 的结果保存到 detections  % 详解: 赋值：将 x(...) 的结果保存到 detections
dst = sqdist(sources, detections);  % 详解: 赋值：将 sqdist(...) 的结果保存到 dst

[a, ass] = bipartiteMatchingIntProg(dst);  % 详解: 执行语句
[a2] = bipartiteMatchingHungarian(dst);  % 详解: 执行语句
assert(isequal(a(:), a2(:)))  % 详解: 调用函数：assert(isequal(a(:), a2(:)))

figure(4); clf  % 详解: 执行语句
bipartiteMatchingDemoPlot(sources, detections, a)  % 详解: 调用函数：bipartiteMatchingDemoPlot(sources, detections, a)

p1 = size(sources, 2);  % 详解: 赋值：将 size(...) 的结果保存到 p1
p2 = size(detections, 2);  % 详解: 赋值：将 size(...) 的结果保存到 p2
nmatch = ceil(0.8*min(p1,p2));  % 详解: 赋值：将 ceil(...) 的结果保存到 nmatch
a2 = bipartiteMatchingIntProg(dst, nmatch);  % 详解: 赋值：将 bipartiteMatchingIntProg(...) 的结果保存到 a2
figure(5); clf  % 详解: 执行语句
bipartiteMatchingDemoPlot(sources, detections, a2)  % 详解: 调用函数：bipartiteMatchingDemoPlot(sources, detections, a2)



ndx = find(label==2);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
sources = x(ndx,:)';  % 赋值：设置变量 sources  % 详解: 赋值：将 x(...) 的结果保存到 sources  % 详解: 赋值：将 x(...) 的结果保存到 sources
ndx = find(label==1);  % 详解: 赋值：将 find(...) 的结果保存到 ndx
detections = x(ndx,:)';  % 赋值：设置变量 detections  % 详解: 赋值：将 x(...) 的结果保存到 detections  % 详解: 赋值：将 x(...) 的结果保存到 detections
dst = sqdist(sources, detections);  % 详解: 赋值：将 sqdist(...) 的结果保存到 dst

p1 = size(sources, 2);  % 详解: 赋值：将 size(...) 的结果保存到 p1
p2 = size(detections, 2);  % 详解: 赋值：将 size(...) 的结果保存到 p2
nmatch = ceil(0.8*min(p1,p2));  % 详解: 赋值：将 ceil(...) 的结果保存到 nmatch
a2 = bipartiteMatchingIntProg(dst, nmatch);  % 详解: 赋值：将 bipartiteMatchingIntProg(...) 的结果保存到 a2
figure(6); clf  % 详解: 执行语句
bipartiteMatchingDemoPlot(sources, detections, a2)  % 详解: 调用函数：bipartiteMatchingDemoPlot(sources, detections, a2)




