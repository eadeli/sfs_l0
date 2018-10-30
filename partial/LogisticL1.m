function [alpha, beta] = LogisticL1(X,y,lambda1,k,k1,k2,tol,M,maxiter)
% Logistic loss + L1 regularizer

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

a = 0; b = 10;
lambda = lambda1;
[alpha, beta] = LogisticPartial(X,y,lambda,0,k1,k2,tol,M,maxiter);
while abs(k-nnz(beta))/k>0.05

    if nnz(beta)>k
        a = lambda1; lambda = (a+b)/2;
    else
        b = lambda1; lambda = (a+b)/2;
    end
    [alpha, beta] = LogisticPartial(X,y,lambda,0,k1,k2,tol,M,maxiter);
end