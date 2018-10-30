function [Fval, eAx] = Fval_L1(A,x,lambda,r)

%Evaluate the function value 
%  F(x) = sum(log(1+exp(Ax)))+lambda*sum(|x|_[r+1:n])

n = max(size(x));
eAx = 1 + exp(A(x,1));
f = sum(log(eAx));
tmp = sort(abs(x));
Fval = f + lambda*sum(tmp(1:n-r)); 




