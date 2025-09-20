function [fuzzy_images] = gen_fuzzy_images(x_train, param, norm_u_k)
% Transform the raw images into fuzzy images

pixel_num = size(x_train, 2);
fuzzy_images = cell(param.rule_num, 1);

for i = 1: param.rule_num
    norm_u_i = norm_u_k(:, i);
    norm_u_i_repmat = repmat(norm_u_i, 1, pixel_num);
    fuzzy_images{i, 1} = x_train.*norm_u_i_repmat;
end

end