function [x, fval, time] = L1_se(A,b,lambda,r,tol,M,maxiter,x0)
% subproblem for minimizing squared error loss + partial regularization wrt beta
% a warm start strategy is employed

tic
Lmin = 1e-8;
Lmax = 1e8;
c = 1e-4;
tau = 1.1;
x = x0;
% Create function handle
if ~isa(A,'function_handle')
  A = @(x,mode) explicitMatrix(A,x,mode);
end
gx = A(A(x,1)-b,2).*2;
tmp = sort(abs(x));
n = length(x);
Fx = norm(b-A(x,1),2)^2+lambda*sum(tmp(1:n-r)); 
Fl = -ones(M,1)*inf; 
Fl(1) = Fx; Fmax = Fx; 
L = 1; err = inf;
iter = 1; nf = 0;

% Main loop
while iter <= maxiter && err > tol
    while 1 == 1
    	xnew = L1_partial_prox(x-gx/L,lambda/L,r);
        tmp = sort(abs(xnew));
        Fxnew = norm(b-A(xnew,1),2)^2+lambda*sum(tmp(1:n-r)); 
        nf = nf + 1;
        if (Fmax - Fxnew < c*norm(xnew-x)^2/2)
            L = tau*L;
        else
            break;
        end
    end
    gxnew = A(A(xnew,1)-b,2).*2;
    dx = xnew - x; 
    dg = gxnew - gx;
    err = norm(L*dx-dg,inf);
    L = max(Lmin,min(Lmax,dx'*dg/norm(dx)^2));
	Fl(mod(iter,M)+1) = Fxnew;
    Fmax = max(Fl);   
    x = xnew;
    gx = gxnew;
    iter = iter + 1;
end
fval = Fxnew;
time = toc;
