function [f,expf] = fun(A,mu,sigma,b,x)

n = size(A,1); expf = zeros(n,1);
tmp = -operator(A,mu,sigma,b,x,1);
I = find(tmp>50); J = find(tmp<=50);
expf(J) = exp(tmp(J));
expf(I) = 1e20;
f = (sum(log(1+expf(J))) + sum(tmp(I)))/n;
