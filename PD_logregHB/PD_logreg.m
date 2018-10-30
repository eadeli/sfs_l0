% aim to solve the problem:
% min e'*f(Ax)/n
% s.t. |x(2:p+1)|_0 <= k,
% where e is the all-ones vector, f(y) = log(1+exp(-y)),
% A = [b1*z1 b1; b2*z2 b2; ... bn*zn bn],
% and bi in {-1,1} and zi in Re^p for i=1, ..., n. 
%        
% by applying penalty method to the problem:
% min e'*f(Ax)/n
% s.t. y - x = 0, |x(2:p+1)|_0 <= k,
%
%------ input ------
%
% Z     - the n x p data matrix
% b     - binary outcomes
% Ztype - `SM' means Z has small or medium number of rows and/or columns; 
%         otherwise, Z is large-scale.
% k     - the desired cardinality (i.e., the number of nonzeros)
% eps   - the tolerance for termination
% maxit - the maximum of number of iterations for running code
%
%------ output ------
%
% x     - approximate sparse solution
% obj   - objective value at x
%

function [x,obj] = PD_logreg(Z,b,Ztype,lambda,k,eps,M,maxit,x0)

global typeA U d tolcg Afull

typeA = Ztype; tolcg = 1e-2;

% Preprocess data
[n,p] = size(Z); 
% U = []; d = [];
% mu = mean(Z);
% sigma = zeros(p,1);
% for (i = 1:p)
%     tmp = Z(:,i) - mu(i);
%     sigma(i) = sqrt(tmp'*tmp/(n-1));
% end
% I0 = find(sigma~=0);
% Z = Z(:,I0); mu = mu(I0); sigma = sigma(I0); 
% ptmp = p;
% p = length(I0); 
% 
% if strcmp(typeA,'SM') % small or medium
%      for i = 1:p
%          %Z(:,i) = (Z(:,i) -mu(i))/sigma(i);
%      end
%      A = [diag(b)*Z];
% else
%      A = sparse(diag(b))*Z;
%      Afull = zeros(n,p);
%      for i = 1:n
%             Afull(i,:) = (A(i,:)- b(i)*mu)./sigma' ;
%      end
% end
% 
% if k > p
%     k = p;
% end
A = -diag(b)*Z;
if ~isa(A,'function_handle')
    A = @(x,mode) explicitMatrix(A,x,mode);
end
% Initialize
z = rand(k,1); I = randperm(p);
x0 = zeros(p,1); x0(I(1:k)) = z;
z = x0;
x = x0; x_old = x; y = x;
rho = 0.1; iter = 1; tol = 1e-2; 

while 1==1 
  best_obj = inf;
  while 1==1
    % solve y 
    y = L2_logistic(A,rho,x,lambda,y,eps,M,maxit);
    % solve x 
    [~,I] = sort(abs(y),'descend');
    x = zeros(p,1); I = I(1:k);
    x(I) = y(I); 
    err = norm(x-x_old,inf)/max(norm(x,'inf'),1);
    [f, eAx] = Fval_L1(A,y);
    obj = f/size(A,1);
    obj = obj + rho*norm(x-y)^2/2+norm(y)^2*(lambda/2);
    res = norm(x-y,inf);
    iter = iter + 1; 
    if err <= tol
      if best_obj - obj > (1e-3)*abs(best_obj) || (best_obj == inf && obj < best_obj) 
        best_obj = obj; best_k = nnz(x); 
        best_x = x; best_res = res; r = 1;
      end
      % reduce the cardinality of best_x by r
      if r >= min(2,best_k+1) || rho > 1; break; end;
      [~,I] = sort(abs(best_x),'descend');
      x = best_x; x(I(best_k-r+1:best_k)) = 0;
      r = r+1; 
    end
    x_old = x;
  end
  if best_obj < obj 
    x = best_x; res = best_res;
  end
  if res < 1e-4 || iter > maxit; break; end;  
  rho = min(sqrt(10)*rho,1e15);
  tol = max(tol/sqrt(10), eps);
  x_old = x;
end

% % Postprocess
% xtmp = zeros(ptmp,1);
% xtmp(I0) = x(1:p)./sigma;
% x  = xtmp(1:ptmp);  

