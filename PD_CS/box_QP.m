% solve the box QP problem via ADM 
%  min x'*Q*x/2 - c'*x
%  s.t. l <= x <= u,
% where Q = A'*A + rho I, by solving the problem 
%  min y'*Q*y/2 - c'*y
%  s.t. y - x = 0,
%       l <= x <= u. 

function [x,z] = box_QP(A,AT,b,c,l,u,rho,eps,x,z,flag)

global typeAAt U d;


if l == -inf && u == inf && rho == 0
  dd = d;
  if strcmp(typeAAt,'I')
     x = c;
  else
    if flag
       x = AT(cg(A,AT,b,0));     
    else
       x = cg(AT,A,c,0);
    end
  end
  return;
end

if l == -inf && u == inf && rho > 0
  dd = d + rho;
  if strcmp(typeAAt,'I') 
     x = (c - AT(A(c))/(1+rho))/rho;
  else
    if flag
       z = cg(A,AT,A(c),rho,z); x = (c - AT(z))/rho;            
    else
       x = cg(AT,A,c,rho,x);
    end
  end
  return;
end

mu = max(0.15*p,50);
theta = rho + mu;
dd = d + theta;
[n_x,m_x] = size(x);
lambda = zeros(n_x,m_x);
x_old = x; y = x;  
err = inf;

while (err > eps)
  % update y
  v = c + lambda + mu*x;
  if strcmp(typeAAt,'I')
     y = (v - AT(A(v))/(1+theta))/theta;
  else 
    if flag 
       z = cg(A,AT,A(v),theta,z); y = (v - AT(z))/theta;  
    else
       y = cg(AT,A,v,theta,y);
    end
  end 
  % update x
  x = min(max(y-lambda/mu,l),u);
  % update lambda
  lambda = lambda - mu*(y-x);
  err = norm(x(:)-x_old(:),'inf');
  x_old = x;
end
