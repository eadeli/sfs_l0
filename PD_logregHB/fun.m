function [fx,gx,hx] = fun(y,A,rho,x)
[m,n] = size(A);
if ~isa(A,'function_handle')
  A = @(x,mode) explicitMatrix(A,x,mode);
end
eAx = 1 + exp(A(y,1));
fx = sum(log(eAx))/m+(y-x)'*(y-x)*(rho/2);
gx = A(1-1./eAx,2)+(y-x)*rho;

g = A'*