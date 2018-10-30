function acc = NonConvexJFSS_CV(Data,lossType,regType,kList,k1List,...
    k2List,Lambdalist,strategy)
% non-convex JFSS with hyper-paramtertuning on k, k1 and k2

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


acc = -inf;
err = 0;
numSamples = 0;
C_Type = 'None';


for cv = 1:K 
    % obtain training data
    Xtr = Data{cv,1};
    Ytr = Data{cv,3};
    % split the training data into K_inner_cv folds
    indices0 = crossvalind('Kfold',size(Xtr,2),K_inner_cv);
    
    data0 = cell(K,4);
    for cv0 = 1:K_inner_cv
        data0{cv0,1} = Xtr(:,indices0 ~= cv0);
        data0{cv0,2} = Xtr(:,indices0 == cv0);
        data0{cv0,3} = Ytr(indices0 ~= cv0);
        data0{cv0,4} = Ytr(indices0 == cv0);
    end
    % try the given parameters one by one
    AccList = []; 
    kBin = []; k1Bin = []; k2Bin = [];
    if strategy > 0
        for j = 1:strategy
            k = kList(ceil(length(kList)*rand));
            k1_temp = ceil(k1List(ceil(length(k1List)*rand))*length(find(data0{cv,3}'==1)));
            k2_temp = ceil(k2List(ceil(length(k2List)*rand))*length(find(data0{cv,3}'==-1)));    
            switch regType                           
                case 'L0'
                    acc_temp = CrossValidationAccuracy(data0,...
                        K_inner_cv,lossType,regType,[],k,k1_temp,...
                        k2_temp,tol,M,maxiter);
                otherwise
                    acc_temp = CrossValidationAccuracy_HPT(data0,...
                        K_inner_cv,lossType,regType,k,k1_temp,...
                        k2_temp,tol,M,maxiter,Lambdalist,K_inner_cv);
            end
            kBin = [kBin;k]; k1Bin = [k1Bin;k1_temp]; k2Bin = [k2Bin;k2_temp];
            AccList = [AccList;acc_temp];        
        end
    else
        for k = kList
            for k1 = k1List
                for k2 = k2List
                    k1_temp = ceil(k1*length(find(data0{cv,3}'==1)));
                    k2_temp = ceil(k1*length(find(data0{cv,3}'==-1)));
                    switch regType
                        case 'L0'
                            acc_temp = CrossValidationAccuracy(data0,...
                                K_inner_cv,lossType,regType,[],k,k1_temp,...
                                k2_temp,tol,M,maxiter);
                        otherwise
                            acc_temp = CrossValidationAccuracy_HPT(data0,...
                                K_inner_cv,lossType,regType,k,k1_temp,...
                                k2_temp,tol,M,maxiter,Lambdalist,K_inner_cv);
                    end
                    kBin = [kBin;k]; k1Bin = [k1Bin;k1_temp]; k2Bin = [k2Bin;k2_temp];
                    AccList = [AccList;acc_temp];
                end
            end
        end
    end
    
    % pick the parameter with the best performance
    [~,imax] = max(AccList);
    switch regType
        case 'L0'
            [alpha, beta] = jfss(Data{cv,1}',Data{cv,3}',lossType,regType,...
                [],kBin(imax),k1Bin(imax),k2Bin(imax),tol,M,maxiter);
        otherwise
            AccList0 = [];
            for i = 1:length(Lambdalist)
                acc_temp = CrossValidationAccuracy(data0,K_inner_cv,...
                    lossType,regType,Lambdalist(i),kBin(imax),k1Bin(imax),...
                    k2Bin(imax),tol,M,maxiter);
                AccList0 = [AccList0;acc_temp];
            end
            % pick the parameter with the best performance
            [~,imax0] = max(AccList0);
            lambda1 = Lambdalist(imax0);
            [alpha, beta] = jfss(Data{cv,1}',Data{cv,3}',lossType,regType,...
            lambda1,kBin(imax),k1Bin(imax),k2Bin(imax),tol,M,maxiter);
    end
    
    numSamples = numSamples+size(Data{cv,4},2);
    err = err+testErr(Data,cv,alpha,beta,C_Type);
end
    
acc = 1-err/numSamples;