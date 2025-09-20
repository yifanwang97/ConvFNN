function [weights_fully_connected, flatten_results_global] = computing_final_weights(final_pooling_results, param, y, weight_local)

% Operations on labels
N = size(y, 1);
number_of_class = max(y);
y_global = -ones(N, number_of_class);   
for i = 1: N
    y_global(i, y(i)) = 1; 
end

y_local = repmat(y_global, param.rule_num, 1);

flatten_results_global = cell2mat(final_pooling_results)';

B = size(flatten_results_global, 2)/param.rule_num; %The total number of elements in all the pooling images of one fuzzy rule
flatten_results_local = zeros(B*param.rule_num, N*param.rule_num);
for i = 1: param.rule_num
    flatten_rule = [];
    for j = 1: param.conv_num
        flatten_rule = [flatten_rule; final_pooling_results{j + (i - 1)*param.conv_num, 1}];
    end
    flatten_results_local((i - 1)*B + 1: i*B, (i - 1)*N + 1: i*N) = flatten_rule;
end
flatten_results_local = flatten_results_local';
clear final_pooling_results

% Computing the final weights
sum1 = param.alpha*flatten_results_global'*flatten_results_global + param.beta*flatten_results_local'*weight_local* flatten_results_local + eye(size(flatten_results_global, 2));
sum2 = param.alpha*flatten_results_global'*y_global + param.beta*flatten_results_local'*weight_local*y_local;
weights_fully_connected = sum1\sum2;


end