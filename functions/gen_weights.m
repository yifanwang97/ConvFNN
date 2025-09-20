function [W, pool_index, h_dim] = gen_weights(param)

% Initialize indices for the convolutional layer
[rf_index, h_dim]=initialize_rf_indices(param);

% Initialize indices for the pooling layer
pool_index=initialize_pooling_indices(param, h_dim);

% Initialize the randomly generated convolutional weights
W = orth_conv_weights(param, h_dim, rf_index);


end

