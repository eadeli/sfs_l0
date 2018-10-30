function [x, fval, time] = L2_logistic(A,rho,z,x0,tol,M,maxiter)

%fprintf(' ************ start the iterations ************\n')

tic

Lmin = 1e-8;
Lmax = 1e8;
c = 1e-4;
tau = 1.1;
x = x0;
m = size(A,1);
% Create function handle
if ~isa(A,'function_handle')
  A = @(x,mode) explicitMatrix(A,x,mode);
end

[Fx,eAx] = Fval_L1(A,x);
Fx = Fx/m+norm(x-z)^2*rho/2;
gx = A(1-1./eAx,2)/m;

Fl = -ones(M,1)*inf; 
Fl(1) = Fx; Fmax = Fx; 
L = 1; err = inf;
iter = 1; nf = 0;

% Main loop
while iter <= maxiter && err > tol
    while 1 == 1
    	xnew = (x*L-gx+z*rho)/(L+rho);
        [Fxnew,eAx] = Fval_L1(A,xnew);
        Fxnew = Fxnew/m+norm(xnew-z)^2*rho/2;
        nf = nf + 1;
        if (Fmax - Fxnew < c*norm(xnew-x)^2/2)
            L = tau*L;
        else
            break;
        end
    end
    gxnew = A(1-1./eAx,2)/m;
    dx = xnew - x; 
    dg = gxnew - gx;
    err = norm(L*dx-dg,inf);
    L = max(Lmin,min(Lmax,dx'*dg/norm(dx)^2));
	Fl(mod(iter,M)+1) = Fxnew;
    Fmax = max(Fl);   
    x = xnew;
    gx = gxnew;
 %   fprintf(' Iter = %3.0d    Fval = %7.3f  err = %10.5f\n', iter, Fxnew, err)
    iter = iter + 1;
end
fval = sum(log(eAx));
time = toc;
% fprintf('\n\n --------------- The computational result ----------------\n\n')
% fprintf(' Iter = %3.0d   nf=%d   fval = %7.3f  nnz=%d   time = %.2f\n', ....
%       iter, nf, fval, nnz(x),toc)
% fprintf('\n ----------------------------------------------------------\n\n')

