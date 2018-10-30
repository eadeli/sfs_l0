% script for testing the sparse logistic regression code PD_logreg

clear all
n = 1000 ; p = 100; 
Xtype = 'SM';
eps =1e-4;
maxit = inf;
I = randperm(n); b = ones(n,1);
b(I(1:n/2)) = -1; X = zeros(n,p);
for (j = 1:n)
    X(j,:) = b(j)*rand + randn(1,p);
end

[x,obj] = PD_logreg(X,b,Xtype,0.1*p,eps,maxit);

