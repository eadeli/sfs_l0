function generate_imbal_toy_dataset ()
    r = [0 0.1 0.2 0.3 0.4 0.5]; % Portion of noisy data of noise
    num_classes = 2;
    num_samples_cp = 100; num_samples_cn = 50;
    num_samples = num_samples_cp + num_samples_cn;
    num_features = 80;
    
    Xs = zeros(num_features, num_samples, length(r));
    Ys = zeros(num_samples, length(r));
    
    for i = 1:length(r)
        [Xs(:,:,i) Ys(:,i)] = generate_data_00 (num_features, num_samples_cp, num_samples_cn, r(i), r(i));
    end
    
    clear i
    save toy_data_imbal_1 % P = 100, N = 50
end

function [X Y] = generate_data_00 (num_features, num_samples_cp, num_samples_cn, r_features, r_samples)
    neg_class = -1; % Can be set to '0' as well
    num_samples = (num_samples_cp + num_samples_cn);
    ncoeff = 0.5;
    %X = zeros(num_features, num_samples);
    %Y = zeros(num_samples, 1);

    d = num_features / 2;
    [U,S,V] = svd(rand(num_features));
    U1 = U(:,1:d);
    X = U1*rand(d,num_samples_cp);
    Y = ones(num_samples_cp,1);

    R = orth(rand(num_features));
    U1 = R*U1;
    X = [X,U1*rand(d,num_samples_cn)];
    Y = [Y;(neg_class)*ones(num_samples_cn,1)];
    
    %adding noise to r ratio of the samples 
    norm_x = sqrt(sum(X.^2,1));
    norm_x = repmat(norm_x,num_features,1);
    gn = norm_x.*randn(num_features,num_samples);
    inds = zeros(1,num_samples, 'logical');
    inds(1:floor(num_samples_cp * r_samples)) = 1;
    inds(num_samples_cp+1:num_samples_cp+floor(num_samples_cn * r_samples)) = 1;
    %inds(randsample(num_samples, floor(num_samples * r_samples))) = 1;
    %inds = rand(1,num_samples) < r_samples;
    X(:,inds) = X(:,inds) + ncoeff*gn(:,inds);

    %adding noise to r ratio of the features 
    %norm_x = sqrt(sum(X.^2,2));
    %norm_x = repmat(norm_x,1,num_samples);
    gn = norm_x.*randn(num_features,num_samples);
    inds = zeros(1,num_features, 'logical');
    inds(1:floor(num_features * r_features)) = 1;
%    inds = rand(1,num_features) < r_features;
    X(inds,:) = X(inds,:) + ncoeff*gn(inds,:);
end