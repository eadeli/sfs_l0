function [acc, BAcc, F1, pred, selected_features, selected_samples] = CrossValidationAccuracy(Data,K,lossType,regType,lambda1,...
    k,k1,k2,tol,M,maxiter,cvs_inner)
% the K-fold cross validation

%------inputs:
% Data: data set
% K: number of folds
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
% acc: K-fold cross validation results

global all_labels
global test_idx_split
global cvs
global num_features
global num_samples

if (isempty(cvs))
    cvss = test_idx_split;
else
    cvss = cvs;
end
if (nargin > 11)
    cvss = cvs_inner;
end

err = 0;
acc = -inf;
numSamples = 0;

C_Type = 'None';
pred = [];

selected_features = zeros(num_features,1);
%ttt=size(Data{1,1}, 2);
selected_samples = zeros(num_samples,1);
cp = classperf (cellstr(num2str(all_labels)));
for cv = 1:K
    % train a model 
    if (strcmp(regType, "GL0"))
        [alpha, beta, groupsW, vvv] = jfss(Data{cv,1},Data{cv,3}',lossType,regType,...
        lambda1,k,k1,k2,tol,M,maxiter);
    else
        [alpha, beta] = jfss(Data{cv,1}',Data{cv,3}',lossType,regType,...
        lambda1,k,k1,k2,tol,M,maxiter);
    end
    % verify that the model satifies the requirement on variable sparsity
    switch lossType
        case 'logisticB'
            numFeatures = nnz(beta(2:end));
        otherwise
            numFeatures = nnz(beta);
    end
    if (numFeatures~=k || length(find(Data{cv,3}'.*alpha==1))~=k1...
            || length(find(Data{cv,3}'.*alpha==-1))~=k2) ...
            && strcmp(regType,'L1') == 0&& strcmp(regType,'GL0') == 0
        %disp('The specified requirement on model sparsity is not satisfied. Model failed!');
        BAcc = -inf;
        F1 = -inf;

        %return;
    end
    % compute the errors in validating such model
    if (strcmp(regType,'GL0'))
        nnn = size(Data{cv,2},3);
        pred0 = zeros(nnn,1);
        pred = zeros(nnn,1);
        w = groupsW(:);
        for j = 1:nnn
             temp = Data{cv,2}(:,:,j);
             pred0(j) = (w'*temp(:)+vvv);
             pred(j) = sign(w'*temp(:)+vvv);
             %pred(j) = sign(trace(W'*X(:,:,j))+v);
        end
        err = err + length(Data{cv,4}) - nnz(pred==Data{cv,4}');
        %selected_features = selected_features + (beta(1:num_features) ~= 0);
        %aaa = 1:num_samples;
        %aaa(cvs{cv,1}) = [];
        %selected_samples(aaa) = selected_samples(aaa) + (alpha(:) ~= 0);
        numSamples = numSamples+nnn;
        classperf (cp, cellstr(num2str(sign(pred0))), cvss{cv, 1});

    else
        [eee, pred0] = testErr(Data,cv,alpha,beta,C_Type);
        %selected_features = selected_features + (beta(1:num_features) ~= 0);
        %aaa = 1:num_samples;
        %aaa(cvs{cv,1}) = [];
        %selected_samples(aaa) = selected_samples(aaa) + (alpha(:) ~= 0);
        err = err+eee;
        pred = [pred; pred0];
        numSamples = numSamples+size(Data{cv,4},2);

        classperf (cp, cellstr(num2str(sign(pred0))), cvss{cv, 1});
    end

end

acc = 1-err/numSamples;
BAcc = (cp.DiagnosticTable(1,1) / (cp.DiagnosticTable(1,1)+cp.DiagnosticTable(2,1)) + cp.DiagnosticTable(2,2) / (cp.DiagnosticTable(2,2)+cp.DiagnosticTable(1,2))) / 2;
F1 = 2 * cp.DiagnosticTable(1,1) / (2 * cp.DiagnosticTable(1,1) + cp.DiagnosticTable(2,1) + cp.DiagnosticTable(1,2));
