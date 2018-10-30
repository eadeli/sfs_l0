% Test nonconvex JFSS-C
close all
clear all
clc

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
k = 40;
% number of positive samples to be selected
k1 = 25;
% number of negative samples to be selected
k2 = 25;
% regularization parameters
lambda1 = .5;
%Lambdalist = [0;0.001;0.005;0.01;0.05;0.1;0.5;1;5];
Lambdalist = [0;0.001;0.01;0.1;1];
% 0: print nothing, 1: print main result, 2: print all info
verbosity = 2; 

global all_labels
global cvs
global num_features
global num_samples
%% tests start here
addpath('JFSS')
addpath('funs')
load toy_data_imbal_1.mat
num_samples = size(Xs,2);
num_features = size(Xs,1);
indices = crossvalind('Kfold',num_samples,K);
% ACC_se_l0 = [];
% ACC_logi_l0 = [];
% ACC_se_partial = [];
% ACC_logi_partial = [];
% ACC_logi_l0l2 = [];

ACC_logiB_l0 = [];
ACC_logiB_partial = [];
ACC_logiB_l0l2 = [];
BACC_logiB_l0 = [];
BACC_logiB_partial = [];
BACC_logiB_l0l2 = [];
F1_logiB_l0 = [];
F1_logiB_partial = [];
F1_logiB_l0l2 = [];

noise_feat = [0, 8, 16, 24, 32, 40];
noise_samp_p = [0, 10, 20, 30, 40, 50];
noise_samp_n = [0, 5, 10, 15, 20, 25];

for noise_lev_idx = 1:length(r)
    % begin a test with specified noise level

    if (verbosity >= 2)
        fprintf ('* noise_lev_idx: %d\n', noise_lev_idx);
    end
    % permute the samples
    X = squeeze(Xs(:,:,noise_lev_idx)); 
    Y = squeeze(Ys(:,noise_lev_idx))';
    all_labels = Y';
%     aa = randperm(num_samples);
%     X(:,:) = X(:,aa);
%     Y(:,:) = Y(:,aa);
    % split the data set into K folds
    cvs = cell(K,1);
    for cv = 1:K
        cvs{cv} = find(indices == cv);
    end

    data = cell(K,4);
    for cv = 1:K
        data{cv,1} = X(:,indices ~= cv);
        data{cv,2} = X(:,indices == cv);
        data{cv,3} = Y(:,indices ~= cv);
        data{cv,4} = Y(:,indices == cv);
    end
    
    % methods without hyper parameter tuning        
%     acc_se_l0    = CrossValidationAccuracy(data,K,'se','L0',...
%         [],k,k1,k2,tol,M,maxiter) 
%     acc_logi_l0  = CrossValidationAccuracy(data,K,'logistic','L0',...
%         [],k,k1,k2,tol,M,maxiter)
     [acc_logiB_l0, BAcc_l0, F1_l0, pred, selected_features, selected_samples] = CrossValidationAccuracy(data,K,'logisticB','L0',...
        [],k,k1,k2,tol,M,maxiter);
    BAcc_l0;
    
    err_feat = 0; err_samp = 0;
    if (noise_lev_idx > 1)
        % only when toy_data_imbal_1 is used
        % in toy_data_imbal_0 data is randomly contaminated by noise
        err_feat = sum(selected_features(1:noise_feat(noise_lev_idx)) > 7) / (noise_feat(noise_lev_idx));
        bbb = sum(selected_samples(1:noise_samp_p(noise_lev_idx)) > 7);
        bbb = bbb + sum(selected_samples(101:100+noise_samp_n(noise_lev_idx)) > 7);% / (noise_samp_n(noise_lev_idx));
        err_samp = bbb / (noise_samp_p(noise_lev_idx) + noise_samp_n(noise_lev_idx));
    end
    
    % methods with hyper parameter tuning    
%     acc_se_partial = CrossValidationAccuracy_HPT(data,K,'se','partial',...
%         k,k1,k2,tol,M,maxiter,Lambdalist,K_inner_cv)
%     acc_logi_partial = CrossValidationAccuracy_HPT(data,K,'logistic',...
%         'partial',k,k1,k2,tol,M,maxiter,Lambdalist,K_inner_cv)
    BAcc_partial = 0; F1_partial = 0; acc_logiB_partial = 0;
    [acc_logiB_partial, BAcc_partial, F1_partial, pred, selected_samples, selected_features] = CrossValidationAccuracy_HPT(data,K,...
        'logisticB','partial',k,k1,k2,tol,M,maxiter,Lambdalist,K_inner_cv);

%     acc_logi_l0l2 = CrossValidationAccuracy_HPT(data,K,'logistic','L0L2',...
%         k,k1,k2,tol,M,maxiter,Lambdalist,K_inner_cv)
    
    BAcc_l0l2 = 0; F1_l0l2 = 0; acc_logiB_l0l2 = 0;
     [acc_logiB_l0l2, BAcc_l0l2, F1_l0l2, pred, selected_samples, selected_features] = CrossValidationAccuracy_HPT(data,K,'logisticB','L0L2',...
         k,k1,k2,tol,M,maxiter,Lambdalist,K_inner_cv);

%     ACC_se_l0 = [ACC_se_l0;acc_se_l0];
%    ACC_logi_l0 = [ACC_logi_l0;acc_logi_l0];
%     ACC_se_partial = [ACC_se_partial;acc_se_partial];
%    ACC_logi_partial = [ACC_logi_partial;acc_logi_partial];
%    ACC_logi_l0l2 = [ACC_logi_l0l2;acc_logi_l0l2];
    ACC_logiB_l0 = [ACC_logiB_l0;acc_logiB_l0];
    ACC_logiB_partial = [ACC_logiB_partial;acc_logiB_partial];
    ACC_logiB_l0l2 = [ACC_logiB_l0l2;acc_logiB_l0l2];
    BACC_logiB_l0 = [BACC_logiB_l0;BAcc_l0];
    BACC_logiB_partial = [BACC_logiB_partial;BAcc_partial];
    BACC_logiB_l0l2 = [BACC_logiB_l0l2;BAcc_l0l2];
    F1_logiB_l0 = [F1_logiB_l0;F1_l0];
    F1_logiB_partial = [F1_logiB_partial;F1_partial];
    F1_logiB_l0l2 = [F1_logiB_l0l2;F1_l0l2];
    
    fprintf ('BAccs: L0 (%f), Partial (%f), L0L2 (%f) - L0 feature errors (%f) - L0 samples errors (%f) \n ', BAcc_l0, BAcc_partial, BAcc_l0l2, err_feat, err_samp);
end

%Resutls = [ACC_se_l0 ACC_logi_l0 ACC_logiB_l0 ACC_se_partial ACC_logi_partial ACC_logiB_partial ACC_logi_l0l2 ACC_logiB_l0l2]'