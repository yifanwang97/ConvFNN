function [conv_results, pooling_results] = conv_pooling(X, W, pool_index)

% convolutional layer
conv_results = W*X;
% square-root pooling operation
pooling_results = sqrt(full(pool_index)*(conv_results.^2));


end
