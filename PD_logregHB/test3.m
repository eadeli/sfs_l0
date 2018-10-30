% Test nonconvex JFSS-C
clear
close all

%% parameter setting
% number of folds in the outter cross validation (CV)
K = 10;
% number of folds in the inner CV for hyper-parameter tuning
K_inner_cv = 5;
% numerical precision
tol = 1e-4;
% size of the bin keeping latest results
M = 5;
% maximum number of iterations
maxiter = 1e6;

% least number of features to be selected
kList = [40 60 80];
% percentage of positive samples to be selected
k1List = [0.5 0.75 1];
% percentage of negative samples to be selected
k2List = [0.5 0.75 1];

Lambdalist = [0.001;0.005;0.01;0.05;0.1;0.5;1;5];
%Lambdalist = [0.001;0.01;0.1;1];
% 0: print nothing, 1: print main result, 2: print all info
verbosity = 2; 
ng = 5

%% tests start here
addpath('JFSS')
addpath('funs')
load toy_data_0.mat
numSamples = size(Xs,2);
indices = crossvalind('Kfold',num_samples,K);
ACC_se_l0 = [];
ACC_logi_l0 = [];
ACC_logiB_l0 = [];
ACC_se_partial = [];
ACC_logi_partial = [];
ACC_logiB_partial = [];
ACC_logi_l0l2 = [];

for noise_lev_idx = 2:length(r)
    % begin a test with specified noise level

    if (verbosity >= 2)
        fprintf ('* noise_lev_idx: %d\n', noise_lev_idx);
    end
    % permute the samples
    X = squeeze(Xs(:,:,noise_lev_idx)); 
    Y = squeeze(Ys(:,noise_lev_idx))';
    aa = randperm(num_samples);
    X(:,:) = X(:,aa);
    Y(:,:) = Y(:,aa);
    % split the data set into K folds
    data = cell(K,4);
    for cv = 1:K
        data{cv,1} = X(:,indices ~= cv);
        data{cv,2} = X(:,indices == cv);
        data{cv,3} = Y(:,indices ~= cv);
        data{cv,4} = Y(:,indices == cv);
    end
    
    acc_logi_partial = NonConvexJFSS_CV(data,'logistic',...
         'partial',kList,k1List,k2List,Lambdalist,ng)
    ACC_logi_partial = [ACC_logi_partial;acc_logi_partial];

    acc_logiB_partial = NonConvexJFSS_CV(data,...
        'logisticB','partial',kList,k1List,k2List,Lambdalist,ng)    
    ACC_logiB_partial = [ACC_logiB_partial;acc_logiB_partial];
    
    

    % methods without hyper parameter tuning        
%     acc_se_l0    = NonConvexJFSS_CV(data,'se','L0',...
%         kList,k1List,k2List,Lambdalist,ng)
%     ACC_se_l0 = [ACC_se_l0;acc_se_l0];
    
    acc_logi_l0  = NonConvexJFSS_CV(data,'logistic','L0',...
        kList,k1List,k2List,Lambdalist,ng)
    ACC_logi_l0 = [ACC_logi_l0;acc_logi_l0];
    
    acc_logiB_l0 = NonConvexJFSS_CV(data,'logisticB','L0',...
        kList,k1List,k2List,Lambdalist,ng)
    ACC_logiB_l0 = [ACC_logiB_l0;acc_logiB_l0];
    
    % methods with hyper parameter tuning    
%     acc_se_partial = NonConvexJFSS_CV(data,'se','partial',...
%         kList,k1List,k2List,Lambdalist,ng)
%     ACC_se_partial = [ACC_se_partial;acc_se_partial];
    
    acc_logi_partial = NonConvexJFSS_CV(data,'logistic',...
        'partial',kList,k1List,k2List,Lambdalist,ng)
    ACC_logi_partial = [ACC_logi_partial;acc_logi_partial];
    
    acc_logi_l0l2 = NonConvexJFSS_CV(data,'logistic','L0L2',...
        kList,k1List,k2List,Lambdalist,ng)
    ACC_logi_l0l2 = [ACC_logi_l0l2;acc_logi_l0l2];
end