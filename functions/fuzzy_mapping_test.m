function [fuzzy_images_test, weight_local_test] = fuzzy_mapping_test(x_test, param, center, delta)

% Select several pixel values with large variances 
std_test = std(x_test,0,1);
[sort_test,index_test] = sort(std_test);
x_test_sub = x_test(:,index_test(end-(param.D_ref-1):end));

% Computing the normalized firing strength
[norm_u_k, weight_local_test] = norm_firing_strength_test(x_test_sub, param, center, delta);

% Generate the fuzzy images
fuzzy_images_test = gen_fuzzy_images(x_test, param, norm_u_k);
end