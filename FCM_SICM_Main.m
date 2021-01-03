%% BEST RUN WITH MATLAB R2018b!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robust Fuzzy C-Means Clustering Algorithm with Adaptive Spatial & Intensity Constraint and Membership Linking for Noise Image Segmentation
% Applied Soft Computing 92 (2020) 106318
% DOI: https://doi.org/10.1016/j.asoc.2020.106318
% Please kindly cite this paper as you like.
%
% This code was solely written by Qingsheng Wang.
%
%% Basically, you can run this code SEVERAL times to acquire the most desired result.
%% It is welcomed to change the following parameters as you like to see what gonna happen.

%% Please tune the parameters according to the instructions in the paper. 
% For example
% For fingerpint.tif in 15% mixed noise, it is best segmented with sigma_d = 2.5 and sigma_r = 5.
% For other images in 15% mixed noise, it is best segmented with sigma_d = 5 and sigma_r = 2 or 2.5.
% to get the best result.

% Inputs:
% density - Mixed noise density
% error - Minimum absolute difference between ath J and (a-1)th J
% cluster_num - Number of clusters
% max_iter - Max iterations
% ==============Parameters for fast bilateral filter================
% sigma_d - geometric spread
% sigma_r - photometric spread;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Intialization
clear all;
close all;
%% Parameters
% Parameters for bilateral filter
sigma_d = 5;
sigma_r = 2.5;
% Parameters for clustering
density = 0.1;
error = 0.001;
cluster_num = 4;
max_iter = 100;
m = 2;
%% Input image
f_uint8 = imread('test4.jpg'); % ff restores input image as uint8
f = double(f_uint8) / 255;
[row, col, depth] = size(f);
N = row * col;
%% Construct mixed noise
f = imnoise(f,'gaussian',0,density);
f = imnoise(f,'salt & pepper',density);
f = imnoise(f,'speckle',density);
figure, imshow(f);
%% Process fast bilateral filtering
f_bilateral = zeros(row, col, depth);
for i = 1 : depth
    f_bilateral(:, :, i) = bilateralFilter(f(:, :, i), [], 0, 1, sigma_d, sigma_r);
end
%% Pixel reshaping
all_pixel = repmat(reshape(f, N, 1, depth), [1 cluster_num 1]);
all_pixel_bi = repmat(reshape(f_bilateral, N, 1, depth), [1 cluster_num 1]);
%% Acquire difference
difference = 20 * abs(all_pixel - all_pixel_bi) + eps;
%% Clustering initialization
% Randomly initialize membership degrees
U = rand(N, cluster_num);
U_col_sum = sum(U, 2);
U = U ./ repmat(U_col_sum, [1 cluster_num]);
U_m = U .^ m;
% Constraint alpha for conventional FCM
alpha = 1 ./ difference;
% Constraint beta for local information
beta = difference;
%% FCM Clustering
for iter = 1 : max_iter
    U_m = repmat(U_m, [1 1 depth]);
    % Update cluster centers by Eq. (25)
    center = sum(U_m .* (alpha .* all_pixel + beta .* all_pixel_bi)) ./ sum(U_m .* (alpha + beta));
    % Compute similarity
    center_rep = repmat(center, [N 1 1]);
    d1 = mean(alpha .* (all_pixel - center_rep) .^ 2, 3);
    d2 = mean(beta .* (all_pixel_bi - center_rep) .^ 2, 3);
    d = d1 + d2;
    % Compute membership linking by Eq. (13)
    membership_linking = repmat(log(sum(U) + 1) .^ 2, [N 1]);
    % Update membership degrees by Eq. (24)
    U_numerator = (d ./ membership_linking) .^ (1 / (m - 1));
    U = U_numerator .* repmat(sum(1 ./ U_numerator, 2), [1, cluster_num]);
    U =  1./ U;
    U_m = U .^ m;
    % Update objective function J by Eq. (20)
    J(iter)=sum(sum(U_m .* d ./ membership_linking));
    fprintf('Iter %d\n', iter);
    % Iteration stopping condition
    if iter > 1 && abs(J(iter) - J(iter - 1)) <= error
        fprintf('Objective function is converged\n');
        break;
    end
    if iter > 1 && iter == max_iter && abs(J(iter) - J(iter - 1)) > error
        fprintf('Objective function is not converged. Max iteration reached\n');
        break;
    end
end
center = uint8(gather(squeeze(center * 255)));
%% Output segmentation result
[~, cluster_indice] = max(U, [], 2);
cluster_indice = reshape(cluster_indice, [row, col]);
% Visualize all labels
FCM_result = Label_image(f_uint8, reshape(cluster_indice, row, col));
figure, imshow(FCM_result);
title('Segmentation result');
% Visualize objective function
figure, plot(J);
title('Objective function J');
