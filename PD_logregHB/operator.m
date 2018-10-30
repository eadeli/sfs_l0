function val = operator(A,mu,sigma,b,x,flag)

global typeA

if flag == 1
    if strcmp(typeA,'SM')
        val = A*x;
    else
    p = size(A,2);
    x(1:p) = x(1:p)./sigma;
    val = (A*x(1:p) - b*(mu*x(1:p) -x(p+1))); 
    end
else
    if strcmp(typeA,'SM')
        val = A'*x;
    else
    p = size(A,2);
    val = A'*x - mu'*b'*x;
    val(1:p) = val(1:p)./sigma;
    val = [val;b'*x];
    end
end
