function[W] = orth_conv_weights(param, h_dim, rf_index)

% Generate the initialized convolutional weights
W_ini = randn(param.conv_num, param.input_ch*param.window_size^2);

% orthogonalize the convolutinal weights
for i = 1:param.conv_num
    W_temp = W_ini(i,:);
    W_temp_ch = [];
    for j = 1:param.input_ch
        W_temp_j = W_temp(1, param.window_size*param.window_size*(j-1)+1:param.window_size*param.window_size*j);
        W_reshape = reshape(W_temp_j, param.window_size, param.window_size);
        W_orth = orth(W_reshape);
        W_orth_reshape = reshape(W_orth, 1, param.window_size*param.window_size);
        W_temp_ch = [W_temp_ch, W_orth_reshape];
    end
    W_dense(i,:) = W_temp_ch;
end


% Expand the dense convolutional weights corresponding to the rf_index.
W_expand=expand_rf(param, h_dim, W_dense);
W=full_size(W_expand, rf_index);


end