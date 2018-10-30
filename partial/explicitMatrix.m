function y = explicitMatrix(A,x,mode)

% Function that is used to create function handles A(x,mode) for 
% matrix multiplication.

switch mode
  case 1
    y = A*x;
  case 2
    y = A'*x;
  otherwise
    error('Unknown mode passed to explicitMatrix in fpc.m');
end

end