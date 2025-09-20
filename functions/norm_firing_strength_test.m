function[norm_u_k, weight_local_test] = norm_firing_strength_test(x_test_sub, param, center, delta)

% Compute the kernel width in Gaussian fuzzy membership function
[data_num, D_ref] = size(x_test_sub);

% Compute the firing strength
for i = 1: param.rule_num
    center_repmat = repmat(center(i, :), data_num, 1);
    delta_repmat = repmat(delta(i, :), data_num, 1);
    % Gaussian fuzzy membership degree
    u_k_d = exp(-(x_test_sub - center_repmat).^2./(2*delta_repmat));
    % Compute the firing strength
    u_k(:, i) = prod(u_k_d, 2);
end

% Compute the normalized firing strength
sum_u_k = sum(u_k,2);
norm_u_k = u_k./repmat(sum_u_k, 1, param.rule_num);

% Construct the local weight for local learning
weight_temp = reshape(norm_u_k, data_num * param.rule_num, 1);
list = 1:data_num*param.rule_num;
weight_local_test = sparse(list, list, weight_temp, data_num*param.rule_num, data_num*param.rule_num);

end