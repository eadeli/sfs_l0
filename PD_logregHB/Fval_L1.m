function [f, eAx] = Fval_L1(A,x)
% Evaluate the function value 
% F(x) = sum(log(1+exp(Ax)))

eAx = 1 + exp(A(x,1));
f = sum(log(eAx)); 