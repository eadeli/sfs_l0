% cg:  solve (theta I + W*W')x = b using conjugate gradient,
% Zhaosong Lu, March 2010
% Department of Mathematics
% Simon Fraser University

function [x, iter] = cg(W,WT,b,theta,x)

global tolcg

[n,m] = size(b);  iter = 0;
maxiter = min(250,.5*n*m);
if (nargin < 5)
   x = zeros(n,m); r = b;
else
   r = b - theta*x - W(WT(x));
end;
rnrm = norm(r,'fro');
p = r;
err = inf;
while (err > tolcg) && (iter <= maxiter)
    Ap = theta*p + W(WT(p));
    pAp = p(:)'*Ap(:);
    alpha = rnrm^2/pAp;
    x = x + alpha*p;
    r = r - alpha*Ap;
    oldrnrm = rnrm; rnrm = norm(r,'fro');
    err = norm(r(:),inf)/max(max(abs(x(:))),1);
    beta = (rnrm/oldrnrm)^2;
    p = r + beta*p; iter = iter + 1;
end
clear r Ap p;

