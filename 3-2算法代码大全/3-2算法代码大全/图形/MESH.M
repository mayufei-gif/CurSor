% 文件: MESH.M
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function h = mesh(x,y,z,c)  % 详解: 绘图：三维网格




user_view = 0;  % 详解: 赋值：计算表达式并保存到 user_view
cax = newplot;  % 详解: 赋值：计算表达式并保存到 cax
fc = get(gca,'color');  % 详解: 赋值：将 get(...) 的结果保存到 fc
if strcmp(lower(fc),'none')  % 详解: 条件判断：if (strcmp(lower(fc),'none'))
    fc = get(gcf,'color');  % 详解: 赋值：将 get(...) 的结果保存到 fc
end  % 详解: 执行语句

if nargin == 1  % 详解: 条件判断：if (nargin == 1)
    hh = surface(x,'FaceColor',fc,'EdgeColor','flat', 'FaceLighting', 'none', 'EdgeLighting', 'flat');  % 详解: 赋值：将 surface(...) 的结果保存到 hh
elseif nargin == 2  % 详解: 条件判断：elseif (nargin == 2)
        if isstr(y), error('Invalid argument.'); end  % 详解: 条件判断：if (isstr(y), error('Invalid argument.'); end)
    [my ny] = size(y);  % 详解: 获取向量/矩阵尺寸
    [mx nx] = size(x);  % 详解: 获取向量/矩阵尺寸
    if mx == my & nx == ny  % 详解: 条件判断：if (mx == my & nx == ny)
        hh = surface(x,y,'FaceColor',fc,'EdgeColor','flat', 'FaceLighting', 'none', 'EdgeLighting', 'flat');  % 详解: 赋值：将 surface(...) 的结果保存到 hh
    else  % 详解: 条件判断：else 分支
        if my*ny == 2  % 详解: 条件判断：if (my*ny == 2)
            hh = surface(x,'FaceColor',fc,'EdgeColor','flat', 'FaceLighting', 'none', 'EdgeLighting', 'flat');  % 详解: 赋值：将 surface(...) 的结果保存到 hh
            set(gca,'View',y);  % 详解: 调用函数：set(gca,'View',y)
            user_view = 1;  % 详解: 赋值：计算表达式并保存到 user_view
        else  % 详解: 条件判断：else 分支
            error('Invalid input arguments.');  % 详解: 调用函数：error('Invalid input arguments.')
        end  % 详解: 执行语句
    end  % 详解: 执行语句
elseif nargin == 3  % 详解: 条件判断：elseif (nargin == 3)
        if isstr(y) | isstr(z), error('Invalid argument.'); end  % 详解: 条件判断：if (isstr(y) | isstr(z), error('Invalid argument.'); end)
    if min(size(y)) == 1 & min(size(z)) == 1  % 详解: 条件判断：if (min(size(y)) == 1 & min(size(z)) == 1)
        hh = surface(x,'FaceColor',fc,'EdgeColor','flat', 'FaceLighting', 'none', 'EdgeLighting', 'flat');  % 详解: 赋值：将 surface(...) 的结果保存到 hh
        set(gca,'View',y);  % 详解: 调用函数：set(gca,'View',y)
        user_view = 1;  % 详解: 赋值：计算表达式并保存到 user_view
    else  % 详解: 条件判断：else 分支
        hh = surface(x,y,z,'FaceColor',fc,'EdgeColor','flat', 'FaceLighting', 'none', 'EdgeLighting', 'flat');  % 详解: 赋值：将 surface(...) 的结果保存到 hh
    end  % 详解: 执行语句
elseif nargin == 4  % 详解: 条件判断：elseif (nargin == 4)
    hh = surface(x,y,z,c,'FaceColor',fc,'EdgeColor','flat', 'FaceLighting', 'none', 'EdgeLighting', 'flat');  % 详解: 赋值：将 surface(...) 的结果保存到 hh
else  % 详解: 条件判断：else 分支
    error('Requires 1, 2, 3, or 4 input arguments.');  % 详解: 调用函数：error('Requires 1, 2, 3, or 4 input arguments.')
end  % 详解: 执行语句
if ~ishold & ~user_view  % 详解: 条件判断：if (~ishold & ~user_view)
    view(3); grid on  % 详解: 执行语句
end  % 详解: 执行语句
if nargout == 1  % 详解: 条件判断：if (nargout == 1)
    h = hh;  % 详解: 赋值：计算表达式并保存到 h
end  % 详解: 执行语句




