function [accuracy] = computing_accuracy(fully_connected_weights, flatten_results_global, y)

predict_Y = flatten_results_global*fully_connected_weights;

data_num = size(y,1);
MissClassificationRate = 0;
for i = 1:data_num
    [dummy, label_expected] = max(predict_Y(i, :));
    if label_expected~=y(i);
        MissClassificationRate = MissClassificationRate + 1;
    end
end
accuracy = 1 - MissClassificationRate/data_num
end