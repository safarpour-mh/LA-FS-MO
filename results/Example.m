%% Feature Ranking on ILPD Subset (Table 1) – Label-Agnostic Multi-Objective Method
% Based on: 
%   - Dataset from "Feature selection for imbalanced data based on neighborhood rough sets" (Table 1)
%   - Methodology from your paper: Shannon entropy + Inverted DBI + Fuzzy Crowding Distance

clear; clc; close all;

% --------------------------------------------------
% Step 1: Define the dataset X (10 samples × 10 features)
% Gender: Male → 1, Female → 0
% --------------------------------------------------
X = [
    72, 1, 3.9, 2.0, 195, 27, 59, 7.3, 2.4, 0.4;   % x1
    46, 1, 1.8, 0.7, 208, 19, 14, 7.6, 4.4, 1.3;   % x2
    26, 0, 0.9, 0.2, 154, 16, 12, 7.0, 3.5, 1.0;   % x3
    29, 0, 0.9, 0.3, 202, 14, 11, 6.7, 3.6, 1.1;   % x4
    17, 1, 0.9, 0.3, 202, 22, 19, 7.4, 4.1, 1.2;   % x5
    72, 1, 2.7, 1.3, 260, 31, 56, 7.4, 3.0, 0.6;   % x6
    64, 1, 0.9, 0.3, 310, 61, 58, 7.0, 3.4, 0.9;   % x7
    38, 0, 0.8, 0.2, 185, 25, 21, 7.0, 3.0, 0.7;   % x8
    46, 0, 0.8, 0.2, 185, 24, 15, 7.9, 3.7, 0.8;   % x9
    53, 0, 0.9, 0.2, 210, 35, 32, 8.0, 3.9, 0.9    % x10
];

[n, d] = size(X); % n=10, d=10

% --------------------------------------------------
% Parameters
% --------------------------------------------------
K_bins = 5;
eps_val = 1e-6;

% --------------------------------------------------
% Step 2: Objective 1 – Discretized Shannon Entropy
% --------------------------------------------------
o1 = zeros(d, 1);
for j = 1:d
    fj = X(:, j);
    if std(fj) < eps_val
        o1(j) = 0;
        continue;
    end
    edges = linspace(min(fj), max(fj), K_bins + 1);
    edges = unique(edges);
    if length(edges) < 2
        o1(j) = 0;
        continue;
    end
    [~, ~, bin_idx] = histcounts(fj, edges);
    actual_bins = length(edges) - 1;
    counts = accumarray(bin_idx, 1, [actual_bins, 1])';
    pk = zeros(1, K_bins);
    pk(1:actual_bins) = counts / n;
    pk(pk == 0) = eps_val;
    o1(j) = -sum(pk .* log2(pk));
end

% --------------------------------------------------
% Step 3: Objective 2 – Inverted DBI (1D k-means, k=2)
% --------------------------------------------------
o2 = zeros(d, 1);
for j = 1:d
    fj = X(:, j);
    if std(fj) < eps_val
        o2(j) = 0;
        continue;
    end
    try
        [idx, C] = kmeans(fj, 2, ...
            'MaxIter', 100, 'Replicates', 5, ...
            'EmptyAction', 'drop', 'Start', 'plus');
        clusters = unique(idx);
        if length(clusters) < 2
            DBIj = inf;
        else
            sigmas = zeros(2, 1);
            for ci = 1:2
                members = fj(idx == clusters(ci));
                if ~isempty(members)
                    sigmas(ci) = std(members);
                else
                    sigmas(ci) = 0;
                end
            end
            c1 = C(1); c2 = C(2);
            dist = abs(c1 - c2) + eps_val;
            DBIj = 0.5 * (sigmas(1) / dist + sigmas(2) / dist);
        end
    catch
        DBIj = inf;
    end
    o2(j) = 1 / (DBIj + eps_val);
end

% --------------------------------------------------
% Step 4: Normalize objectives to [0,1]
% --------------------------------------------------
o1_norm = (o1 - min(o1)) / (max(o1) - min(o1) + eps_val);
o2_norm = (o2 - min(o2)) / (max(o2) - min(o2) + eps_val);

% --------------------------------------------------
% Step 5: Fuzzy Crowding-Distance Ranking
% --------------------------------------------------
[~, sort1] = sort(o1_norm, 'descend');
[~, sort2] = sort(o2_norm, 'descend');

Delta_p1 = inf(d,1); Delta_m1 = inf(d,1);
Delta_p2 = inf(d,1); Delta_m2 = inf(d,1);

for i = 1:d
    pos1 = find(sort1 == i);
    if pos1 > 1
        Delta_m1(i) = o1_norm(i) - o1_norm(sort1(pos1 - 1));
    end
    if pos1 < d
        Delta_p1(i) = o1_norm(sort1(pos1 + 1)) - o1_norm(i);
    end

    pos2 = find(sort2 == i);
    if pos2 > 1
        Delta_m2(i) = o2_norm(i) - o2_norm(sort2(pos2 - 1));
    end
    if pos2 < d
        Delta_p2(i) = o2_norm(sort2(pos2 + 1)) - o2_norm(i);
    end
end

% Fuzzy spread parameters
sigma1 = (max(o1_norm) - min(o1_norm)) / 10 + eps_val;
sigma2 = (max(o2_norm) - min(o2_norm)) / 10 + eps_val;

% Fuzzy membership
mu_p1 = 1 - exp(-(Delta_p1 / sigma1).^2);
mu_m1 = 1 - exp(-(Delta_m1 / sigma1).^2);
mu_p2 = 1 - exp(-(Delta_p2 / sigma2).^2);
mu_m2 = 1 - exp(-(Delta_m2 / sigma2).^2);

% Final fuzzy crowding score
CDfuzzy = (mu_p1 + mu_m1 + mu_p2 + mu_m2) / 4;

% --------------------------------------------------
% Step 6: Rank features (descending)
% --------------------------------------------------
[CD_sorted, rank_idx] = sort(CDfuzzy, 'descend');

% --------------------------------------------------
% Step 7: Save results for paper writing
% --------------------------------------------------
results = struct();
results.X = X;
results.feature_names = {...
    'Age', 'Gender', 'Total_bilirubin', 'Direct_bilirubin', ...
    'Total_proteins', 'Albumin', 'A/G_ratio', 'SGPT', 'SGOT', 'Alkphos'};
results.o1_raw = o1;
results.o2_raw = o2;
results.o1_norm = o1_norm;
results.o2_norm = o2_norm;
results.CDfuzzy = CDfuzzy;
results.feature_rank = rank_idx;
results.CD_sorted = CD_sorted;

save('ILPD_Example_Results.mat', 'results');

% --------------------------------------------------
% Step 8: Display results
% --------------------------------------------------
fprintf('Dataset X (10×10):\n');
disp(X);

fprintf('\nFeature Ranking (Best → Worst):\n');
for i = 1:d
    feat_name = results.feature_names{rank_idx(i)};
    fprintf('Rank %2d: %15s | CD = %.4f\n', i, feat_name, CD_sorted(i));
end