function [x, fval, time] = L1_logisticB(A,C,lambda,r,tol,M,maxiter,x0)
% subproblem for minimizing logistic loss (the classifier contains a bias term) + partial regularization wrt beta
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

[Fx,eAx] = Fval_L1B(A,C,x,lambda,r);
gx = A(1-1./eAx,2);

Fl = -ones(M,1)*inf; 
Fl(1) = Fx; Fmax = Fx; 
L = 1; err = inf;
iter = 1; nf = 0;

% Main loop
while iter <= maxiter && err > tol
    while 1 == 1
    	xnew = L1_partial_prox(x-gx/L,lambda/L,r);
        [Fxnew,eAx] = Fval_L1B(A,C,xnew,lambda,r);
        nf = nf + 1;
        if (Fmax - Fxnew < c*norm(xnew-x)^2/2)
            L = tau*L;
        else
            break;
        end
    end
    gxnew = A(1-1./eAx,2);
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
fval = sum(log(eAx));
time = toc;

