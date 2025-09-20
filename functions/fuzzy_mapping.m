function [fuzzy_images, weight_local, center, delta] = fuzzy_mapping(x_train, param)
% weight_local and fuzzy_images are for computing the fully_connected weights
% center and delta are parameters in fuzzy membership function for testing datasets

% Select several pixel values with large variances 
std_train = std(x_train,0,1);
[sort_train,index_train] = sort(std_train);
x_train_sub = x_train(:,index_train(end-(param.D_ref-1):end));

% Use fuzzy c-means clustering compute the fuzzy membership degree of each data point in each cluster
[center,u_n_k,obj_fcn] = fcm(x_train_sub, param.rule_num);

% Computing the normalized firing strength
[norm_u_k, weight_local, delta] = norm_firing_strength(x_train_sub, param, center, u_n_k);

% Generate the fuzzy images
fuzzy_images = gen_fuzzy_images(x_train, param, norm_u_k);


end