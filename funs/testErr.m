function [z, pred] = testErr(Data,cv,alpha,beta,C_Type)   
% evaluate the performance of the trained JFSS model

%------inputs:
% Data: the given data set
% cv: the index of data subset used for the test
% alpha: indictor for the selection of samples
% beta: the resulting weight vector of the JFSS model
% C_Type: the type of classifier

%------outputs:
% z: the number of errors occurred in the validation dataset

if strcmp(C_Type,'None')
    if length(beta)>size(Data{cv,1},1)
        % the model contains a bias term
        Xt = [ones(size(Data{cv,2},2),1) Data{cv,2}'];
        z = nnz(sign(Xt*beta)-Data{cv,4}');
        pred = Xt*beta;
        return
    else
        z = nnz(sign(Data{cv,2}'*beta)-Data{cv,4}');
        pred = Data{cv,2}'*beta;
        return
    end
    
end

if strcmp(C_Type,'LogisticRegression')
    addpath('partial')
    % create new training data 
    m = sum(alpha);
    X = Data{cv,1}'; 
    X(alpha==0,:) = []; 
    y = Data{cv,3}'; 
    y(alpha==0) = [];
    
    if length(beta)>size(X,2)  
        % the model contains a bias term
        X(:,beta(2:end)==0) = [];
        X = [X ones(m,1)];
    else
        X(:,beta==0) = [];
    end

    % train a classifier with the above data
    % note that the initial pt is obtained from the raw JFSS results
    signedX = -spdiags(y,0,m,m)*X;
    tA = signedX;
    if ~isa(tA,'function_handle')
        tA = @(x,mode) explicitMatrix(tA,x,mode);
    end
    
    if length(beta)>size(Data{cv,1},1)
        beta0 = beta(2:end); 
        beta0 = nonzeros(beta0);%beta0(abs(beta0)>1e-8); 
        beta0 = [beta0;beta(1)];
    else
        beta0 = nonzeros(beta); 
    end

    [w,~] = L1_logistic(tA,0,0,1e-4,5,1e4,beta0);
    
    % compute the errors in the validation data
    Xt = Data{cv,2}'; 
    if length(beta)>size(Data{cv,1},1)
        Xt(:,beta(2:end)==0) = [];
        Xt = [Xt ones(size(Data{cv,2},2),1)];
    else
        Xt(:,beta==0) = [];
    end
    
    yt = Data{cv,4}';
    z = nnz(sign(Xt*w)-yt);
	pred = Xt*w;
    
    rmpath('partial')
    return
end