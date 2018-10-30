function [alpha, beta, groupsW, vvv] = jfss(X,y,LossType,RegType,lambda1,k,k1,k2,tol,M,maxiter)
% a wrapper for current methods

%------inputs:
% X: data matrix
% y: label vector
% lossType: logistic, logisticB, or sum of square
% regType: L1, L0L2, partial L1 or L0
% lambda1: regularization parameter
% k: sparsity level of beta
% k1: number of positive instances
% k2: number of negative instance
% tol: precision
% M: length of the bin holding the latest results
% maxiter: maximum number of iterations


%------outputs:
% alpha: variable for sample seleection
% beta: variable for feature selection

% metheds
switch LossType
    case 'logistic'
        switch RegType
            case 'partial'
%                 disp('Model: Logistic loss with partial regularizers')
                addpath('partial')
                [alpha, beta] = LogisticPartial(X,y,lambda1,k,k1,k2,tol,M,maxiter);
                rmpath('partial')
            case 'L0'
                %disp('Model: Logistic loss with L0 regularizers')
                addpath('PD_logreg')
                [alpha, beta] = LogisticL0(X,y,k,k1,k2,tol,M,maxiter);
                rmpath('PD_logreg')
            case 'GL0'
                disp('Model: Logistic loss with Group L0 regularizers --- NOT IMPLEMENTED ---')
                %addpath('group_PD_logregBHB')
                %[alpha, beta] = groupLogisticL0L2(X,y,lambda1,k,k1,k2,tol,M,maxiter);
                %rmpath('group_PD_logregBHB')
            case 'L0L2'
                %disp('Model: Logistic loss with L0L2 regularizers')
                addpath('PD_logregHB')
                [alpha, beta] = LogisticL0L2(X,y,lambda1,k,k1,k2,tol,M,maxiter);
                rmpath('PD_logregHB')
            case 'L1'
                %disp('Model: Logistic loss with L1 regularizers')
                addpath('partial')
                [alpha, beta] = LogisticL1(X,y,lambda1,k,k1,k2,tol,M,maxiter);
                rmpath('partial')
        end
        
    case 'logisticB'
        switch RegType
            case 'partial'
                %disp('Model: Logistic loss with partial regularizers, add an intercept on the classifer')
                addpath('partial')
                [alpha, beta] = LogisticPartialB(X,y,lambda1,k,k1,k2,tol,M,maxiter);
                rmpath('partial')
            case 'L0'
                %disp('Model: Logistic loss with L0 regularizers, add an intercept on the classifer')
                addpath('PD_logregB')
                [alpha, beta] = LogisticL0(X,y,k,k1,k2,tol,maxiter);
                rmpath('PD_logregB')
            case 'GL0'
                %disp('Model: Logistic loss with Group L0 regularizers')
                addpath('group_PD_logregBHB')
                [alpha, beta, groupsW, vvv] = groupLogisticL0L2(X,y,lambda1,k,k1,k2,tol,maxiter);
                rmpath('group_PD_logregBHB')
            case 'L1'
                %disp('Model: Logistic loss with L1 regularizers')
                addpath('partial')
                [alpha, beta] = LogisticL1B(X,y,lambda1,k,k1,k2,tol,M,maxiter);
                rmpath('partial')
                
            case 'L0L2'
                %disp('Model: Logistic loss with L0L2 regularizers')
                addpath('PD_logregBHB')
                [alpha, beta] = LogisticL0L2(X,y,lambda1,k,k1,k2,tol,maxiter);
                rmpath('PD_logregBHB')
        end
    case 'se'
        switch RegType
            case 'partial'
                %disp('Model: sum of squared error loss with partial regularizers')
                addpath('partial')
                [alpha, beta] = SePartial(X,y,lambda1,k,k1,k2,tol,M,maxiter);
                rmpath('partial')
            case 'L0'
                %disp('Model: sum of squared error loss with L0 regularizers')
                addpath('PD_CS')
                [alpha, beta] = SeL0(X,y,k,k1,k2,tol,maxiter);
                rmpath('PD_CS')
            case 'L1'
                %disp('Model: sum of squared error loss with L1 regularizers')
                addpath('partial')
                [alpha, beta] = SeL1(X,y,lambda1,k,k1,k2,tol,M,maxiter);
                rmpath('partial')
        end
end
                
       
   