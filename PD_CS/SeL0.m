function [alpha, beta] = SeL0(X,y,k,k1,k2,tol,maxiter)
% sum of squared error + L0 regularizer

%------inputs:
% k: sparsity level of beta
% k1: number of positive instances
% k2: number of negative instance
% tol: precision
% maxiter: maximum number of iterations

%------outputs:
% alpha for sample selection
% beta for feature selection

[m,n] = size(X);
alpha0 = ones(m,1);
beta0 = zeros(n,1);
I1 = find(y==1); 
I2 = find(y==-1);
loss = abs(y-X*beta0);
loss = loss.^2;
obj0 = sum(loss);
k1 = min(k1,length(I1));
k2 = min(k2,length(I2));

err = inf; 
itx = 0;
while err>tol && itx<maxiter
    itx = itx+1;
    A = X(alpha0>1e-8,:);
    b = y(alpha0>1e-8);
    [beta,~] = PD_CS(A,A','F',b,-inf,inf,k,tol,maxiter,beta0);
    loss = abs(y-X*beta);
    [~,ixtemp] = sort(loss(I1));
    alpha = zeros(m,1);
    i1 = ixtemp(1:k1);
    alpha(I1(i1)) = 1;
    [~,ixtemp] = sort(loss(I2));
    i2 = ixtemp(1:k2);
    alpha(I2(i2)) = 1;
    %err = max(norm(alpha-alpha0,'inf'),norm(beta-beta0,'inf'));
    loss = loss.^2;
    obj = sum(loss.*alpha);
    err = min(abs(obj-obj0)/abs(obj),abs(obj));
    alpha0 = alpha;        
    beta0 = beta;
    obj0 = obj;
end