% aim to solve the problem:
% min  |Ax - b|^2/2
% s.t. l <= x <= u, |x|_0 <= k,
% by applying AL method to the problem:
% min  |Ay - b|^2/2 
% s.t. x - y = 0, 
%      l <= x <= u, |x|_0 <= k,
%      l <= y <= u.
%
%------ input ------
%
% A     - the n x p sample matrix or the corresponding function handle 
% AT    - empty or the function handle for transpose of A
% Atype - type of the matrix A: `I' means A*A' = I;  `F' means 
%         p > n, otherwise, p <= n.
% b     - the n x 1 observation vector
% l     - the lower bound on the solution
% u     - the upper bound on the solution
% k     - the desired cardinality (i.e., the number of nonzeros)
% eps   - the tolerance for termination
% maxit - the maximum of number of iterations for running code
%
%------ output ------
%
% x     - approximate sparse solution
% obj   - objective value at x
%

function [x,obj] = PD_CS(A,AT,Atype,b,l,u,k,eps,maxit,x0)

global typeAAt U d tolcg

tolcg = 1e-4; 

if find(upper(Atype)=='I')
    typeAAt = 'I';
else
    typeAAt = '';
end

if (~isempty(find(upper(Atype)=='F'))) || (p > n)
    flag = 1;
else
    flag = 0;
end
% if A is a function handle, check presence of AT,
if isa(A, 'function_handle') && ~isa(AT,'function_handle')
  error(['The function handle for transpose of A is missing']);
end 

% if A is a matrix, create function handles for multiplication by A and A'.

if ~isa(A, 'function_handle')
  AT = @(x) A'*x;
  A = @(x) A*x;
end

% Initialize
c = AT(b);
n  = length(b); [n_x,m_x] =size(c); p = n_x*m_x;
U = []; d = [];
z = zeros(n,1);
% I = randperm(p);
% x = zeros(p,1);
% x(I(1:k)) = rand(k,1);
x = x0;
iter = 1; 
x_old = x; y = x; 
rho = 1e-1;
tol = 1e-2;

while 1==1 
    best_obj = inf; 
    while 1==1 
        % solve y
        [y,z] = box_QP(A,AT,b,c+rho*x,l,u,rho,1e-4,y,z,flag);
        % solve x
        x = min(max(y,l),u);
        val = (x-y).*(x-y) - y.*y;
        [tmp,I] = sort(val(:),'ascend');
        tmp = zeros(n_x*m_x,1);
        tmp(I(1:k)) = x(I(1:k));
        x = reshape(tmp,n_x,m_x);
        err = norm(x(:)-x_old(:),inf)/max(max(abs(x(:))),1);
        obj = norm(A(y),'fro')^2/2 - c(:)'*y(:) + rho*norm(x-y,'fro')^2/2;
        res = norm(x(:)-y(:),inf);       
        iter = iter + 1;       
        if err <=tol 
            if best_obj - obj > (1e-3)*abs(best_obj) || (best_obj == inf && obj < best_obj) 
                best_obj = obj; best_x = x; best_res = res; 
                best_k = nnz(x); r = 1;
            end
            % reduce the cardinality of best_x by r
            if r >= 2 || rho > 1; break; end;
            x = x(:);
            [tmp,I] = sort(abs(x),'descend');
            x(I(best_k-r+1:best_k)) = 0;
            x = reshape(x,n_x,m_x);
            r = r+1; 
        end
        x_old = x;
    end
    if obj > best_obj
        x = best_x; res = best_res;
    end
    if res < 1e-4 || iter > maxit; break; end;
    rho = min(sqrt(10)*rho,1e8);
    tol = max(tol/sqrt(10), eps);
    x_old = x;
end
obj = norm(A(x)-b,'fro')^2/2 ;


