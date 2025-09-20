function [final_pooling_results] = stacked_conv_pooling(fuzzy_images, W, pool_index, param)

fuzzy_conv_results = cell(param.rule_num, 1);
fuzzy_pooling_results = cell(param.rule_num, 1);

for i = 1: param.rule_num
    [fuzzy_conv_results{i, 1}, fuzzy_pooling_results{i, 1}] = conv_pooling(fuzzy_images{i, 1}', W{i, 1}, pool_index{i, 1});
end

conv_results = cell(param.rule_num * param.conv_num, param.layer_num);
pooling_results = cell(param.rule_num * param.conv_num, param.layer_num);

image_size_s2 = size(fuzzy_pooling_results{1, 1}, 1)/param.conv_num;  

for i = 1:param.rule_num
    for j = 1: param.conv_num
        conv_results{j + (i - 1) * param.conv_num, 1} = fuzzy_conv_results{i, 1}(1 + (j - 1) * image_size_s2: j * image_size_s2, :);
        pooling_results{j + (i - 1) * param.conv_num, 1} = fuzzy_pooling_results{i, 1}(1 + (j - 1) * image_size_s2: j * image_size_s2, :);
    end
end

% From two layers
for i = 1: param.rule_num*param.conv_num
    for j = 2:param.layer_num
        [conv_results{i, j}, pooling_results{i, j}] = conv_pooling(pooling_results{i, j-1}, W{i, j}, pool_index{i, j});
    end
end

% The pooling results in the final layer
final_pooling_results = pooling_results(:, param.layer_num);


end