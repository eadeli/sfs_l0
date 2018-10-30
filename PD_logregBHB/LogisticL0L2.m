function [alpha, beta1] = LogisticL0L2(X,y,lambda,k,k1,k2,tol,maxiter)
% Logistic loss (the classifier contains a bias term) + L0

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
beta0 = [1;zeros(n,1)];
signedX = -spdiags(y,0,m,m)*X;
tA = signedX;
if ~isa(tA,'function_handle')
    tA = @(x,mode) explicitMatrix(tA,x,mode);
end
I1 = find(y==1); 
I2 = find(y==-1);   
eAx = 1 + exp(tA(beta0(2:(end)),1)-beta0(1)*y);
loss = log(eAx);
obj0 = sum(loss)/nnz(alpha0)+lambda*norm(beta0)^2/2;
k1 = min(k1,length(I1));
k2 = min(k2,length(I2));

err = inf; 
itx = 0;
while err>tol && itx<maxiter
    itx = itx+1;
    A = X(alpha0>1e-8,:);
    b = y(alpha0>1e-8);
    [beta1,~] = PD_logreg(A,b,lambda,'SM',k,tol,maxiter,[beta0(2:end);beta0(1)]);
    eAx = 1 + exp(tA(beta1(2:(end)),1)-beta1(1)*y);
    loss = log(eAx);
    [~,ixtemp] = sort(loss(I1));
    alpha = zeros(m,1);
    i1 = ixtemp(1:k1);
    alpha(I1(i1)) = 1;
    [~,ixtemp] = sort(loss(I2));
    i2 = ixtemp(1:k2);
    alpha(I2(i2)) = 1;
    %err = max(norm(alpha-alpha0,'inf'),norm(beta1-beta0,'inf'));
    obj = sum(loss.*alpha)/(k1+k2)+lambda*norm(beta1)^2/2;
    err = min(abs(obj-obj0)/abs(obj),abs(obj));
    alpha0 = alpha;        
    beta0 = beta1;
    obj0 = obj;
end
