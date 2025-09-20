function [W, pool_index]  = gen_whole_weights(param)

W = cell(param.rule_num * param.conv_num, param.layer_num);
pool_index = cell(param.rule_num * param.conv_num, param.layer_num);

% To make the results reproducible, we use a random seed 0;
randn('state', 0);

temp = param.conv_num;
for i = 1: param.layer_num
    for j = 1: param.rule_num
        if i == 1
            % Generate the weights in one convolutional layer
            param.conv_num = temp;
            [W{j, i}, pool_index{j, i}, h_dim] = gen_weights(param);
        else
            param.conv_num = 1;
            param.input_ch = 1;
            for k = 1: temp
                [W{k + temp * (j - 1), i}, pool_index{k + temp * (j - 1), i}, h_dim] = gen_weights(param);
            end        
        end
       
    end
    param.image_size = h_dim;
end

end