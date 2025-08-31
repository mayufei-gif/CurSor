% 文件: PLOT3.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

%空间曲线图
% PLOT3(x,y,z,s)其中x,y,z为向量，每一组分量代表一个数据点。
%                    s为表示颜色、连线和标记选择的字符串
%例如 曲线x=exp(-0.2*z)*cos(0.5*pi*z),y=exp(-0.2*z)*sin(0.5*pi*z), 0<z<20。
%
%  clear;close;z=0:0.1:20;r=exp(-0.2*z);th=0.5*pi*z;
%  x=r.*cos(th);y=r.*sin(th);plot3(x,y,z,'r');
%
%PLOT3  Plot lines and points in 3-D space.
%   PLOT3() is a three-dimensional analogue of PLOT().
% 
%   PLOT3(x,y,z), where x, y and z are three vectors of the same length,
%   plots a line in 3-space through the points whose coordinates are the
%   elements of x, y and z.
% 
%   PLOT3(X,Y,Z), where X, Y and Z are three matrices of the same size,  % 中文: plot3（x，y，z），其中x，y和z是相同大小的三个矩阵，|||绘制从X，Y和Z的列获得的几行。||| plot3（x，y，z，s），其中s是由|||制成的1、2或3个字符串绘图命令下列出的字符。
%   plots several lines obtained from the columns of X, Y and Z.  % 中文: Plot3（X1，Y1，Z1，S1，X2，Y2，Z2，S2，S2，X3，Y3，Z3，S3，S3，...）结合了图|||由（x，y，z，s）四个图定义，其中x，y和z是|||向量或矩阵和S是字符串。 |||示例：螺旋：||| t = 0：pi/50：10*pi;
% 
%   Various line types, plot symbols and colors may be obtained with
%   PLOT3(X,Y,Z,s) where s is a 1, 2 or 3 character string made from  % 中文: plot3（sin（t），cos（t），t）;
%   the characters listed under the PLOT command.  % 中文: plot3将手柄的列向量返回到线对象，一个|||每行处理。 x，y，z三元组或x，y，z，s Quads可以是|||然后是参数/值对，以指定附加|||线的属性。 |||另请参见情节，线，轴，视图，网格，冲浪。 |||如果替换了NextPlot轴属性（关闭），则Plot3重置|||除位置以外的所有轴属性，其默认值和|||删除所有轴儿童（线，补丁，文本，表面和|||图像对象）。
% 
%   PLOT3(x1,y1,z1,s1,x2,y2,z2,s2,x3,y3,z3,s3,...) combines the plots
%   defined by the (x,y,z,s) fourtuples, where the x's, y's and z's are
%   vectors or matrices and the s's are strings.
% 
%   Example: A helix:
% 
%       t = 0:pi/50:10*pi;
%       plot3(sin(t),cos(t),t);
% 
%   PLOT3 returns a column vector of handles to LINE objects, one
%   handle per line. The X,Y,Z triples, or X,Y,Z,S quads, can be 
%   followed by parameter/value pairs to specify additional 
%   properties of the lines.
%
%   See also PLOT, LINE, AXIS, VIEW, MESH, SURF.

%-------------------------------
%   Additional details:
%
%
%   If the NextPlot axis property is REPLACE (HOLD is off), PLOT3 resets 
%   all axis properties, except Position, to their default values, and
%   deletes all axis children (line, patch, text, surface, and
%   image objects).

%   Copyright (c) 1984-98 by The MathWorks, Inc.  % 中文: Mathworks，Inc。版权（C）1984-98
%   $Revision: 5.3 $  $Date: 1997/11/21 23:34:01 $  % 中文: $修订：5.3 $ $日期：1997/11/21 23:34:01 $ |||函数[f wf zwf] = bgf（c，b）|||函数f = boundnetf（g1，g2）||| g（g〜 = 0）= 1;
%   Built-in function.


