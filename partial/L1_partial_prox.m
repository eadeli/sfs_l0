function x = L1_partial_prox(a,lambda,r)

% compute the partial L1 prox subproblem 
% min \|x-a\|^2/2 + lambda*sum(|x|_[r+1:n])

n = max(size(a));
x = a;
[~,I]= sort(abs(a));
I = I(1:n-r);
x(I) = wthresh(a(I),'s',lambda);
end