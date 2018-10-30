function [Fval, eAx] = Fval_L1B(A,C,x,lambda,r)

%Evaluate the function value 
%  F(x) = sum(log(1+exp(Ax+C)))+lambda*sum(|x|_[r+1:n])

n = max(size(x));
eAx = 1 + exp(A(x,1)+C);
f = sum(log(eAx));
tmp = sort(abs(x));
Fval = f + lambda*sum(tmp(1:n-r)); 




