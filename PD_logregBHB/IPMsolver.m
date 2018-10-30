% solve the following problem via IPM 
%  min e'*f(Ax)/n + |x-c|^2*(rho/2)
% where e is the all-ones vector, f(x) = log(1+exp(-x)),

function [x,d] = IPMsolver(A,lambda,mu,sigma,b,c,rho,x,d)
global typeA Afull
eps = 1e-4;
[n,p] = size(A);
iter = 1; 
[f,expf] = fun(A,mu,sigma,b,x);
f = f + 0.5*rho*norm(x-c)^2+0.5*lambda*norm(x)^2;
itx = 0;
while (itx<1e5)
    itx = itx+1;
    [g,D0] = grad_hess(A,lambda,mu,sigma,b,c,rho,x,expf);
    err = norm(g,'inf');
    if err < eps 
        break;
    end
    if strcmp(typeA,'SM')
        tmp = zeros(n,p);
        for i = 1:n
            tmp(i,:) = A(i,:)*sqrt(D0(i));
        end
        if n > p
            hess = tmp'*tmp;
            for i = 1:p 
                hess(i,i) = hess(i,i) + rho+lambda;
            end
                d = -hess\g;
        else
           hess = tmp*tmp';
            for i = 1:n
                hess(i,i) = hess(i,i) + rho+lambda;
            end
            d = -(1/rho)*g + (1/rho)*(tmp'*(hess\(tmp*g)));
        end
    else 
       Pinv =zeros(p+1,1); 
       tmp = zeros(n,p);
       for i = 1:n
            tmp(i,:) = Afull(i,:)*sqrt(D0(i)) ;
            tmp(i,:) = tmp(i,:).^2;
       end
       Pinv(1:p) = 1./(sum(tmp,1) +rho)';
       Pinv(p+1) = 1/(sum(D0)+rho);
       d = PCG(A,mu,sigma,b,D0,rho,g,Pinv,1e-3,inf,d);      
    end
    delta = g'*d;
    [x,f,expf] = linesearch(A,lambda,mu,sigma,b,c,rho,x,d,delta,f);
    iter = iter + 1;
end

     
function [g,D0] = grad_hess(A,lambda,mu,sigma,b,c,rho,x,expz)
n = size(A,1);
tmp = 1/n./(ones(n,1)+1./expz);
g = -operator(A,mu,sigma,b,tmp,2) + rho*(x-c)+lambda*x; 
D0 = 1/n./(2*ones(n,1) + expz + 1./expz);


function [x_new,f_new,expf] = linesearch(A,lambda,mu,sigma,b,c,rho,x,d,delta,f)
alpha = 0.01;
beta = 0.5;
s = 1;
itx = 0;
while (itx<1e5)
    itx = itx+1;
    x_new = x + s*d;
    [f_new,expf] = fun(A,mu,sigma,b,x_new);
    f_new = f_new + 0.5*rho*norm(x_new-c)^2+0.5*lambda*norm(x_new)^2;
    if f_new <= f + alpha*s*delta
        break;
    end
    s = beta*s;
end


function x = PCG(A,mu,sigma,b,D0,rho,g,Pinv,eps,maxiter,x)
iter = 0; r = operator(A,mu,sigma,b,D0.*operator(A,mu,sigma,b,x,1),2) + rho*x + g; 
y = Pinv.*r; p = -y;

while (1)
    iter = iter+1;
    z = operator(A,mu,sigma,b,D0.*operator(A,mu,sigma,b,p,1),2) + rho*p;
    theta = y'*r/(p'*z);
    x = x + theta*p;
    err = norm(r)/norm(g);
    if (err<=eps || iter>maxiter)
        break;
    end
    r_new = r + theta*z;
    y_new = Pinv.*r_new;
    m = y_new'*r_new/(y'*r);
    r = r_new; y =y_new;
    p = -y + m*p;
 end         