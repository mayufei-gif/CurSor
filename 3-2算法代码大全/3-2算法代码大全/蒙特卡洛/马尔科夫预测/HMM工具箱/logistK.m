% 文件: logistK.m
% 说明: 自动添加的注释占位，请根据需要补充。
% 生成: 2025-08-31 23:06
% 注释: 本文件头由脚本自动添加

function [beta,post,lli] = logistK(x,y,w,beta)  % 详解: 函数定义：logistK(x,y,w,beta), 返回：beta,post,lli



error(nargchk(2,4,nargin));  % 详解: 调用函数：error(nargchk(2,4,nargin))

debug = 0;  % 详解: 赋值：计算表达式并保存到 debug
if debug>0,  % 详解: 条件判断：if (debug>0,)
  h=figure(1);  % 详解: 赋值：将 figure(...) 的结果保存到 h
  set(h,'DoubleBuffer','on');  % 详解: 调用函数：set(h,'DoubleBuffer','on')
end  % 详解: 执行语句

[d,nx] = size(x);  % 详解: 获取向量/矩阵尺寸
[k,ny] = size(y);  % 详解: 获取向量/矩阵尺寸

if k < 2,  % 详解: 条件判断：if (k < 2,)
  error('Input y must encode at least 2 classes.');  % 详解: 调用函数：error('Input y must encode at least 2 classes.')
end  % 详解: 执行语句
if nx ~= ny,  % 详解: 条件判断：if (nx ~= ny,)
  error('Inputs x,y not the same length.');  % 详解: 调用函数：error('Inputs x,y not the same length.')
end  % 详解: 执行语句

n = nx;  % 详解: 赋值：计算表达式并保存到 n

sumy = sum(y,1);  % 详解: 赋值：将 sum(...) 的结果保存到 sumy
if abs(1-sumy) > eps,  % 详解: 条件判断：if (abs(1-sumy) > eps,)
  sumy = sum(y,1);  % 详解: 赋值：将 sum(...) 的结果保存到 sumy
  for i = 1:k, y(i,:) = y(i,:) ./ sumy; end  % 详解: for 循环：迭代变量 i 遍历 1:k, y(i,:) = y(i,:) ./ sumy; end
end  % 详解: 执行语句
clear sumy;  % 详解: 执行语句

if nargin < 3,  % 详解: 条件判断：if (nargin < 3,)
  w = ones(1,n);  % 详解: 赋值：将 ones(...) 的结果保存到 w
end  % 详解: 执行语句

w = w / max(w);  % 详解: 赋值：计算表达式并保存到 w

if nargin < 4,  % 详解: 条件判断：if (nargin < 4,)
  beta = 1e-3*rand(d,k);  % 详解: 赋值：计算表达式并保存到 beta
  beta(:,k) = 0;  % 详解: 执行语句
else  % 详解: 条件判断：else 分支
  if sum(beta(:,k)) ~= 0,  % 详解: 条件判断：if (sum(beta(:,k)) ~= 0,)
    error('beta(:,k) ~= 0');  % 详解: 调用函数：error('beta(:,k) ~= 0')
  end  % 详解: 执行语句
end  % 详解: 执行语句

stepsize = 1;  % 详解: 赋值：计算表达式并保存到 stepsize
minstepsize = 1e-2;  % 详解: 赋值：计算表达式并保存到 minstepsize

post = computePost(beta,x);  % 详解: 赋值：将 computePost(...) 的结果保存到 post
lli = computeLogLik(post,y,w);  % 详解: 赋值：将 computeLogLik(...) 的结果保存到 lli

for iter = 1:100,  % 详解: for 循环：迭代变量 iter 遍历 1:100,
  vis(x,y,beta,lli,d,k,iter,debug);  % 详解: 调用函数：vis(x,y,beta,lli,d,k,iter,debug)

  [g,h] = derivs(post,x,y,w);  % 详解: 执行语句

  if rcond(h) < eps,  % 详解: 条件判断：if (rcond(h) < eps,)
    for i = -16:16,  % 详解: for 循环：迭代变量 i 遍历 -16:16,
      h2 = h .* ((1 + 10^i)*eye(size(h)) + (1-eye(size(h))));  % 详解: 赋值：计算表达式并保存到 h2
      if rcond(h2) > eps, break, end  % 详解: 条件判断：if (rcond(h2) > eps, break, end)
    end  % 详解: 执行语句
    if rcond(h2) < eps,  % 详解: 条件判断：if (rcond(h2) < eps,)
      warning(['Stopped at iteration ' num2str(iter) ...  % 详解: 执行语句
               ' because Hessian can''t be conditioned']);  % 详解: 执行语句
      break  % 详解: 跳出循环：break
    end  % 详解: 执行语句
    h = h2;  % 详解: 赋值：计算表达式并保存到 h
  end  % 详解: 执行语句

  lli_prev = lli;  % 详解: 赋值：计算表达式并保存到 lli_prev

  while stepsize >= minstepsize,  % 详解: while 循环：当 (stepsize >= minstepsize,) 为真时迭代
    step = stepsize * (h \ g);  % 详解: 赋值：计算表达式并保存到 step
    beta2 = beta;  % 详解: 赋值：计算表达式并保存到 beta2
    beta2(:,1:k-1) = beta2(:,1:k-1) - reshape(step,d,k-1);  % 详解: 调用函数：beta2(:,1:k-1) = beta2(:,1:k-1) - reshape(step,d,k-1)

    post2 = computePost(beta2,x);  % 详解: 赋值：将 computePost(...) 的结果保存到 post2
    lli2 = computeLogLik(post2,y,w);  % 详解: 赋值：将 computeLogLik(...) 的结果保存到 lli2

    if lli2 > lli,  % 详解: 条件判断：if (lli2 > lli,)
      post = post2; lli = lli2; beta = beta2;  % 详解: 赋值：计算表达式并保存到 post
      break  % 详解: 跳出循环：break
    end  % 详解: 执行语句

    stepsize = 0.5 * stepsize;  % 详解: 赋值：计算表达式并保存到 stepsize
  end  % 详解: 执行语句

  if 1-exp(lli/n) < 1e-2, break, end  % 详解: 条件判断：if (1-exp(lli/n) < 1e-2, break, end)

  dlli = (lli_prev-lli) / lli;  % 详解: 赋值：计算表达式并保存到 dlli
  if abs(dlli) < 1e-3, break, end  % 详解: 条件判断：if (abs(dlli) < 1e-3, break, end)

  if stepsize < minstepsize, brea, end  % 详解: 条件判断：if (stepsize < minstepsize, brea, end)

  if lli < lli_prev,  % 详解: 条件判断：if (lli < lli_prev,)
    warning(['Stopped at iteration ' num2str(iter) ...  % 详解: 执行语句
             ' because the log likelihood decreased from ' ...  % 详解: 执行语句
             num2str(lli_prev) ' to ' num2str(lli) '.' ...  % 详解: 执行语句
            ' This may be a bug.']);  % 详解: 执行语句
    break  % 详解: 跳出循环：break
  end  % 详解: 执行语句
end  % 详解: 执行语句

if debug>0,  % 详解: 条件判断：if (debug>0,)
  vis(x,y,beta,lli,d,k,iter,2);  % 详解: 调用函数：vis(x,y,beta,lli,d,k,iter,2)
end  % 详解: 执行语句

function post = computePost(beta,x)  % 详解: 执行语句
  [d,n] = size(x);  % 详解: 获取向量/矩阵尺寸
  [d,k] = size(beta);  % 详解: 获取向量/矩阵尺寸
  post = zeros(k,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 post
  bx = zeros(k,n);  % 详解: 赋值：将 zeros(...) 的结果保存到 bx
  for j = 1:k,  % 详解: for 循环：迭代变量 j 遍历 1:k,
    bx(j,:) = beta(:,j)'*x;   % 调用函数：bx  % 详解: 执行语句  % 详解: 执行语句
  end  % 详解: 执行语句
  for j = 1:k,  % 详解: for 循环：迭代变量 j 遍历 1:k,
    post(j,:) = 1 ./ sum(exp(bx - repmat(bx(j,:),k,1)),1);  % 详解: 调用函数：post(j,:) = 1 ./ sum(exp(bx - repmat(bx(j,:),k,1)),1)
  end  % 详解: 执行语句
  
function lli = computeLogLik(post,y,w)  % 详解: 执行语句
  [k,n] = size(post);  % 详解: 获取向量/矩阵尺寸
  lli = 0;  % 详解: 赋值：计算表达式并保存到 lli
  for j = 1:k,  % 详解: for 循环：迭代变量 j 遍历 1:k,
    lli = lli + sum(w.*y(j,:).*log(post(j,:)+eps));  % 详解: 赋值：计算表达式并保存到 lli
  end  % 详解: 执行语句
  if isnan(lli),  % 详解: 条件判断：if (isnan(lli),)
    error('lli is nan');  % 详解: 调用函数：error('lli is nan')
  end  % 详解: 执行语句

function [g,h] = derivs(post,x,y,w)  % 详解: 函数定义：derivs(post,x,y,w), 返回：g,h

  [k,n] = size(post);  % 详解: 获取向量/矩阵尺寸
  [d,n] = size(x);  % 详解: 获取向量/矩阵尺寸

  g = zeros(d,k-1);  % 详解: 赋值：将 zeros(...) 的结果保存到 g
  for j = 1:k-1,  % 详解: for 循环：迭代变量 j 遍历 1:k-1,
    wyp = w .* (y(j,:) - post(j,:));  % 详解: 赋值：计算表达式并保存到 wyp
    for ii = 1:d,  % 详解: for 循环：迭代变量 ii 遍历 1:d,
      g(ii,j) = x(ii,:) * wyp';   % 调用函数：g  % 详解: 执行语句  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句
  g = reshape(g,d*(k-1),1);  % 详解: 赋值：将 reshape(...) 的结果保存到 g

  h = zeros(d*(k-1),d*(k-1));  % 详解: 赋值：将 zeros(...) 的结果保存到 h
  for i = 1:k-1,  % 详解: for 循环：迭代变量 i 遍历 1:k-1,
    wt = w .* post(i,:) .* (1 - post(i,:));  % 详解: 赋值：计算表达式并保存到 wt
    hii = zeros(d,d);  % 详解: 赋值：将 zeros(...) 的结果保存到 hii
    for a = 1:d,  % 详解: for 循环：迭代变量 a 遍历 1:d,
      wxa = wt .* x(a,:);  % 详解: 赋值：计算表达式并保存到 wxa
      for b = a:d,  % 详解: for 循环：迭代变量 b 遍历 a:d,
        hii_ab = wxa * x(b,:)';  % 赋值：设置变量 hii_ab  % 详解: 赋值：计算表达式并保存到 hii_ab  % 详解: 赋值：计算表达式并保存到 hii_ab
        hii(a,b) = hii_ab;  % 详解: 执行语句
        hii(b,a) = hii_ab;  % 详解: 执行语句
      end  % 详解: 执行语句
    end  % 详解: 执行语句
    h( (i-1)*d+1 : i*d , (i-1)*d+1 : i*d ) = -hii;  % 详解: 执行语句
  end  % 详解: 执行语句
  for i = 1:k-1,  % 详解: for 循环：迭代变量 i 遍历 1:k-1,
    for j = i+1:k-1,  % 详解: for 循环：迭代变量 j 遍历 i+1:k-1,
      wt = w .* post(j,:) .* post(i,:);  % 详解: 赋值：计算表达式并保存到 wt
      hij = zeros(d,d);  % 详解: 赋值：将 zeros(...) 的结果保存到 hij
      for a = 1:d,  % 详解: for 循环：迭代变量 a 遍历 1:d,
        wxa = wt .* x(a,:);  % 详解: 赋值：计算表达式并保存到 wxa
        for b = a:d,  % 详解: for 循环：迭代变量 b 遍历 a:d,
          hij_ab = wxa * x(b,:)';  % 赋值：设置变量 hij_ab  % 详解: 赋值：计算表达式并保存到 hij_ab  % 详解: 赋值：计算表达式并保存到 hij_ab
          hij(a,b) = hij_ab;  % 详解: 执行语句
          hij(b,a) = hij_ab;  % 详解: 执行语句
        end  % 详解: 执行语句
      end  % 详解: 执行语句
      h( (i-1)*d+1 : i*d , (j-1)*d+1 : j*d ) = hij;  % 详解: 执行语句
      h( (j-1)*d+1 : j*d , (i-1)*d+1 : i*d ) = hij;  % 详解: 执行语句
    end  % 详解: 执行语句
  end  % 详解: 执行语句

function vis (x,y,beta,lli,d,k,iter,debug)  % 详解: 函数定义：vis(x,y,beta,lli,d,k,iter,debug)

  if debug<=0, return, end  % 详解: 条件判断：if (debug<=0, return, end)

  disp(['iter=' num2str(iter) ' lli=' num2str(lli)]);  % 详解: 调用函数：disp(['iter=' num2str(iter) ' lli=' num2str(lli)])
  if debug<=1, return, end  % 详解: 条件判断：if (debug<=1, return, end)

  if d~=3 | k>10, return, end  % 详解: 条件判断：if (d~=3 | k>10, return, end)

  figure(1);  % 详解: 调用函数：figure(1)
  res = 100;  % 详解: 赋值：计算表达式并保存到 res
  r = abs(max(max(x)));  % 详解: 赋值：将 abs(...) 的结果保存到 r
  dom = linspace(-r,r,res);  % 详解: 赋值：将 linspace(...) 的结果保存到 dom
  [px,py] = meshgrid(dom,dom);  % 详解: 执行语句
  xx = px(:); yy = py(:);  % 详解: 赋值：将 px(...) 的结果保存到 xx
  points = [xx' ; yy' ; ones(1,res*res)];  % 详解: 赋值：计算表达式并保存到 points
  func = zeros(k,res*res);  % 详解: 赋值：将 zeros(...) 的结果保存到 func
  for j = 1:k,  % 详解: for 循环：迭代变量 j 遍历 1:k,
    func(j,:) = exp(beta(:,j)'*points);  % 调用函数：func  % 详解: 调用函数：func(j,:) = exp(beta(:,j)'*points)  % 详解: 调用函数：func(j,:) = exp(beta(:,j)'*points); % 调用函数：func % 详解: 调用函数：func(j,:) = exp(beta(:,j)'*points)
  end  % 详解: 执行语句
  [mval,ind] = max(func,[],1);  % 详解: 统计：最大/最小值
  hold off;  % 详解: 执行语句
  im = reshape(ind,res,res);  % 详解: 赋值：将 reshape(...) 的结果保存到 im
  imagesc(xx,yy,im);  % 详解: 调用函数：imagesc(xx,yy,im)
  hold on;  % 详解: 执行语句
  syms = {'w.' 'wx' 'w+' 'wo' 'w*' 'ws' 'wd' 'wv' 'w^' 'w<'};  % 详解: 赋值：计算表达式并保存到 syms
  for j = 1:k,  % 详解: for 循环：迭代变量 j 遍历 1:k,
    [mval,ind] = max(y,[],1);  % 详解: 统计：最大/最小值
    ind = find(ind==j);  % 详解: 赋值：将 find(...) 的结果保存到 ind
    plot(x(1,ind),x(2,ind),syms{j});  % 详解: 调用函数：plot(x(1,ind),x(2,ind),syms{j})
  end  % 详解: 执行语句
  pause(0.1);  % 详解: 调用函数：pause(0.1)





