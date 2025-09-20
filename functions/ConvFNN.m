function [test_accuracy] = ConvFNN(param, x_train, x_test, y_train, y_test)
%% for the training dataset
% fuzzy_mapping
[fuzzy_images, weight_local, center, delta] = fuzzy_mapping(x_train, param);
 
% Generate the whole weights and pool_index in the ConvFNN
[W, pool_index] = gen_whole_weights(param);

% Computing process of stacked convolutional and pooling layers
[final_pooling_results] = stacked_conv_pooling(fuzzy_images, W, pool_index, param);

% Compute the weights in the output layer
[weights_fully_connected, flatten_results_global] = computing_final_weights(final_pooling_results, param, y_train, weight_local);

% Computing the accuracy of the training set
train_accuracy = computing_accuracy(weights_fully_connected, flatten_results_global, y_train);

%% for the testing dataset
% fuzzy mapping, the paprameters in fuzzy memebership function is the same with the parameters in training sets
[fuzzy_images_test, weight_local_test] = fuzzy_mapping_test(x_test, param, center, delta);

% computing the final pooling results using the W computed through training sets
[final_pooling_results_test] = stacked_conv_pooling(fuzzy_images_test, W, pool_index, param);

% process the final pooling results
flatten_results_global_test = cell2mat(final_pooling_results_test)';

% Computing the accuracy of the testing set
test_accuracy = computing_accuracy(weights_fully_connected, flatten_results_global_test, y_test);
end



