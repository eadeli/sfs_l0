function [alpha, beta] = LogisticPartial(X,y,lambda1,k,k1,k2,tol,M,maxiter)
% Logistic loss + partial regularizer

%------inputs:
% lambda1: regularization parameter
% k: sparsity level of beta
% k1: number of positive instances
% k2: number of negative instance
% tol: precision
% M: length of the bin holding the latest results
% maxiter: maximum number of iterations

%------outputs:
% alpha for sample selection
% beta for feature selection

[m,n] = size(X);
alpha0 = ones(m,1);
beta0 = zeros(n,1);
signedX = -spdiags(y,0,m,m)*X;
tA = signedX;
if ~isa(tA,'function_handle')
    tA = @(x,mode) explicitMatrix(tA,x,mode);
end
I1 = find(y==1); 
I2 = find(y==-1); 
eAx = 1 + exp(tA(beta0,1));
loss = log(eAx);
obj0 = sum(loss)/nnz(alpha0);
k1 = min(k1,length(I1));
k2 = min(k2,length(I2));

err = inf; 
iter = 0;
while err>tol && iter <maxiter
    A = signedX(alpha0>1e-8,:);
    if ~isa(A,'function_handle')
        A = @(x,mode) explicitMatrix(A,x,mode);
    end
    [beta,~] = L1_logistic(A,lambda1*nnz(alpha0),k,tol,M,maxiter,beta0);
    eAx = 1 + exp(tA(beta,1));
    loss = log(eAx);
    [~,ixtemp] = sort(loss(I1));
    alpha = zeros(m,1);
    i1 = ixtemp(1:k1);
    alpha(I1(i1)) = 1;
    [~,ixtemp] = sort(loss(I2));
    i2 = ixtemp(1:k2);
    alpha(I2(i2)) = 1;
    [temp,~] = sort(abs(beta));
    obj = sum(loss.*alpha)/(k1+k2)+sum(temp(1:(n-k)))*lambda1;
    err = min(abs(obj-obj0)/abs(obj),abs(obj));
    %err = max([norm(alpha-alpha0,'inf');norm(beta-beta0,'inf')]);
    alpha0 = alpha;        
    beta0 = beta;
    obj0 = obj;
    iter = iter+1;
end