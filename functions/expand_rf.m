function [W] = expand_rf (param, h_dim, collapsed_W)

W = zeros(h_dim^2*param.conv_num, size(collapsed_W,2));

for a = 1: param.conv_num
    W((a-1)*h_dim^2+1:a*h_dim^2, :) = repmat(collapsed_W(a,:), h_dim^2, 1);
end
