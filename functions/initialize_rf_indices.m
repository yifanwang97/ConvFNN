function [rf_index, h_dim] = initialize_rf_indices (param)

% Create index for overlaping receptive fields
h_dim = length(1:param.step:param.image_size-param.window_size+1);

rf_index = zeros(h_dim^2,param.image_size^2);

temp = zeros(param.image_size, param.image_size);

temp(1:param.window_size, 1:param.window_size) = ones(param.window_size);


n = 0;

for a = 0:h_dim-1
    for b = 0:h_dim-1
         n = n + 1; 
         curr_rf = circshift(temp,[a*param.step,b*param.step]);
         rf_index(n,:) = reshape(curr_rf(1:param.image_size,1:param.image_size),param.image_size^2,1);
    end
end

rf_index = repmat (rf_index, param.conv_num, 1);
% num_windows = h_dim^2 * param.conv_num;

rf_index = repmat(rf_index,1,param.input_ch);

rf_index = sparse(logical(rf_index));

end
