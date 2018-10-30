% script for testing the compressed sensing code PD_CS

clear all
close all
scale = 4;
% n is the original signal length
n = 2^12/scale;

% k is number of observations to make
k = 2^10/scale;

% number of spikes to put down
n_spikes = 160/scale;

randn('seed',20)
rand('seed',20)

% random +/- 1 signal
f = zeros(n,1);
q = randperm(n);
f(q(1:n_spikes)) = sign(randn(n_spikes,1));

% measurement matrix
disp('Creating measurement matrix...');
R = randn(k,n);

% orthonormalize rows
R = orth(R')';

if n == 8192  
   % in this case, we load a precomputed
   % matrix to save some time
   load Rmatrix_2048_8192.mat
end
%
disp('Finished creating matrix');

hR = @(x) R*x;
hRt = @(x) R'*x;

% noisy observations
sigma = 0.01; 
y = hR(f) + sigma*randn(k,1);

l = -inf;
u = inf;
eps = 1e-3;
x0 = hRt(y);
[xsol,obj_sol] = PD_CS(hR,hRt,'f',y,-inf,inf,n_spikes,eps,inf);

% Plot results
figure(1)
scrsz = get(0,'ScreenSize');
set(1,'Position',[10 scrsz(4)*0.1 0.9*scrsz(3)/2 3*scrsz(4)/4])
subplot(2,1,1)
plot(f,'LineWidth',1.1)
top = max(f(:));
bottom = min(f(:));
v = [0 n+1 bottom-0.05*(top-bottom)  top+0.05*((top-bottom))];
set(gca,'FontName','Times')
set(gca,'FontSize',14)
title(sprintf('Original (n = %g,p = %g,number of nonzeros = %g)',n,k,n_spikes))
axis(v)

subplot(2,1,2)
plot(xsol,'LineWidth',1.1)
set(gca,'FontName','Times')
set(gca,'FontSize',14)
axis(v)
title(sprintf('PD method ( n = %g, p = %g, number of nonzeros = %g, MSE = %5.3g)',...
    k,n,nnz(xsol),(1/n)*norm(xsol-f)^2))


