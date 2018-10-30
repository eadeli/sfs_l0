function [acc, BAcc, F1, pred, selected_samples, selected_features] = CrossValidationAccuracy_HPT(Data,K,lossType,regType,...
    k,k1,k2,tol,M,maxiter,Lambdalist,K_inner_cv)
% the K-fold cross validation with hyper paramter tuning

%------inputs:
% Data: data set
% K: number of folds
% lossType: logistic, logisticB, or sum of square
% regType: L1, L0L2, partial L1 or L0
% k: sparsity level of beta
% k1: number of positive instances
% k2: number of negative instance
% tol: precision
% M: length of the bin holding the latest results
% maxiter: maximum number of iterations
% m: fold of cross validation
% Lambdalist: the range of hyperparameter
% K_inner_cv: the inner cross validation fold

%------outputs:
% acc: m-fold cross validation results

% Seperate entire data into K folds
% For each fold seperate the training data again into subfolds
% Use cross validation on the subfolds to learn good hyperparameters
% With these hyperparameter build a model on the training data of that fold
% Test the model on the test data
% Repeat on next fold

global num_samples
global num_features
global all_labels
global test_idx_split
global cvs

acc = -inf;
err = 0;
numSamples = 0;

C_Type = 'None';
pred = [];

selected_samples = zeros(num_samples,1);
selected_features = zeros(num_features,1);
cp = classperf (cellstr(num2str(all_labels)));
for cv = 1:K 
        % obtain training data
        Xtr = Data{cv,1};
        Ytr = Data{cv,3};
        % split the training data into K_inner_cv folds
        indices0 = crossvalind('Kfold',size(Xtr,2),K_inner_cv);
        
        data0 = cell(K_inner_cv,4);
        for cv0 = 1:K_inner_cv
            data0{cv0,1} = Xtr(:,indices0 ~= cv0);
            data0{cv0,2} = Xtr(:,indices0 == cv0);
            data0{cv0,3} = Ytr(indices0 ~= cv0);
            data0{cv0,4} = Ytr(indices0 == cv0);
        end
        cvs_inner = cell(K_inner_cv,1);
        for cv0 = 1:K_inner_cv
            cvs_inner{cv0} = find(indices0 == cv0)';
        end

        % try the given parameters one by one
        AccList = []; 
        for i = 1:length(Lambdalist)
            [acc_temp, BAcc_temp] = CrossValidationAccuracy(data0,K_inner_cv,lossType,...
                regType,Lambdalist(i),k,k1,k2,tol,M,maxiter,cvs_inner);
            AccList = [AccList;BAcc_temp];
        end
        % pick the parameter with the best performance
        [~,imax] = max(AccList);
        lambda1 = Lambdalist(imax);
        [alpha, beta] = jfss(Data{cv,1}',Data{cv,3}',lossType,regType,...
            lambda1,k,k1,k2,tol,M,maxiter);
        
        % verify that the model satifies the requirement on variable sparsity
        switch lossType
            case 'logisticB'
                numFeatures = nnz(beta(2:end));
            otherwise
                numFeatures = nnz(beta);
        end
        if (numFeatures~=k || length(find(Data{cv,3}'.*alpha==1))~=k1...
                || length(find(Data{cv,3}'.*alpha==-1))~=k2) ...
                && strcmp(regType,'L1') == 0
            %disp('The specified requirement on model sparsity is not satisfied. Model failed!');
            BAcc = -inf; F1 = -inf;
            %return;    
        end       
        numSamples = numSamples+size(Data{cv,4},2);
        [eee, pred] = testErr(Data,cv,alpha,beta,C_Type);
        err = err+eee;
        classperf (cp, cellstr(num2str(sign(pred))), cvs{cv, 1});

        %selected_samples (cvs_trn{cv}(alpha ~= 0)) = selected_samples (cvs_trn{cv}(alpha ~= 0)) + 1;
        %selected_features (cvs_trn{cv}(beta ~= 0)) = selected_samples (cvs_trn{cv}(beta ~= 0)) + 1;
end
    
acc = 1-err/numSamples;
BAcc = (cp.DiagnosticTable(1,1) / (cp.DiagnosticTable(1,1)+cp.DiagnosticTable(2,1)) + cp.DiagnosticTable(2,2) / (cp.DiagnosticTable(2,2)+cp.DiagnosticTable(1,2))) / 2;
F1 = 2 * cp.DiagnosticTable(1,1) / (2 * cp.DiagnosticTable(1,1) + cp.DiagnosticTable(2,1) + cp.DiagnosticTable(1,2));
