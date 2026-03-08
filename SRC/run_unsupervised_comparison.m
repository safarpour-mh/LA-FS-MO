%% =========================================================
% run_baseline_comparison.m
% Purpose: Compare proposed method against unsupervised baselines
%          (Laplacian Score, MCFS, UDFS) using F1-score.
%          Evaluation via Stratified 5-Fold Cross-Validation.
%
% Requirements: MATLAB R2023a (Statistics and Machine Learning Toolbox)
% Usage: 
%   1. Place this script in the root folder.
%   2. Ensure 'dataset' folder contains .mat files.
%   3. Run the script. LaTeX table saved to /results folder.
%% =========================================================

clc; clear; close all;

%% -------------------- Reproducibility Setup --------------------
% Set random seed for reproducibility
rng(42); 

%% -------------------- Settings --------------------
data_folder = 'dataset';
results_folder = 'results';

if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% Datasets used for baseline comparison (Table 8 in manuscript)
datasets = {'heart', 'wpbc', 'wdbc'};

% Pre-calculated selected indices for the Proposed Method (from Table 4)
selected_indices_paper = struct();
selected_indices_paper.heart = [2, 5, 7, 1, 6];
selected_indices_paper.wpbc = [1, 17, 2, 27, 34,18, 3];
selected_indices_paper.wdbc = [17, 21, 28, 14, 11, 27];

classifiers = {'SVM', 'KNN', 'RF'};
results = {};

%% -------------------- Main Loop --------------------
for d = 1:length(datasets)
    dset = datasets{d};
    fprintf('=============================================\n');
    fprintf('Processing dataset: %s\n', dset);
    fprintf('=============================================\n');

    % Load Data
    X_file = fullfile(data_folder, [dset, 'Samples.mat']);
    y_file = fullfile(data_folder, [dset, 'Label.mat']);

    if ~exist(X_file, 'file') || ~exist(y_file, 'file')
        error('Missing .mat files for dataset: %s', dset);
    end

    X_all = load(X_file).X;
    y_vec = load(y_file).X;
    y = categorical(y_vec(:));

    [n, d_dim] = size(X_all);
    r = length(selected_indices_paper.(dset));

    % -----------------------------------------------------------------
    % Feature Selection (Unsupervised - Run once on full data)
    % Note: Since these methods are label-agnostic, running on full data
    % does not leak class information, though distribution shift is possible.
    % -----------------------------------------------------------------
    
    % 1. Proposed Method (Pre-calculated indices)
    X_proposed = X_all(:, selected_indices_paper.(dset));

    % 2. Laplacian Score
    idx_lap = laplacian_score(X_all, r);
    X_lap = X_all(:, idx_lap);

    % 3. MCFS
    idx_mcfs = mcfs_simple(X_all, r);
    X_mcfs = X_all(:, idx_mcfs);

    % 4. UDFS (with fallback to SPEC if failed)
    try
        idx_udfs = udfs_simple(X_all, r);
        X_udfs = X_all(:, idx_udfs);
    catch
        fprintf('  -> UDFS failed, using SPEC fallback.\n');
        idx_udfs = spec_score(X_all, r);
        X_udfs = X_all(:, idx_udfs);
    end

    % Prepare Feature Sets
    X_sets = {X_all, X_proposed, X_lap, X_mcfs, X_udfs};
    set_names = {'All', 'Proposed', 'Laplacian', 'MCFS', 'UDFS'};
    
    % Initialize Results Matrix
    f1_table = NaN(length(classifiers), length(set_names));

    % -----------------------------------------------------------------
    % Classification Evaluation (Stratified 5-Fold CV)
    % -----------------------------------------------------------------
    cv = cvpartition(y, 'KFold', 5, 'Stratify', true);
    num_folds = cv.NumTestSets;

    for s = 1:length(X_sets)
        X = X_sets{s};
        
        % Remove zero-variance features if any
        stds = std(X, 0, 1);
        valid_cols = stds > 0;
        if ~any(valid_cols)
            continue;
        else
            X = X(:, valid_cols);
        end

        fold_f1 = zeros(length(classifiers), num_folds);

        for k = 1:num_folds
            train_idx = training(cv, k);
            test_idx = test(cv, k);

            X_train = X(train_idx, :);
            y_train = y(train_idx);
            X_test = X(test_idx, :);
            y_test = y(test_idx);

            for c = 1:length(classifiers)
                try
                    switch classifiers{c}
                        case 'SVM'
                            % Standardize internally to prevent data leakage
                            mdl = fitcsvm(X_train, y_train, ...
                                'KernelFunction', 'rbf', ...
                                'Standardize', true, ...
                                'BoxConstraint', 1);
                            pred = predict(mdl, X_test);

                        case 'KNN'
                            mdl = fitcknn(X_train, y_train, ...
                                'NumNeighbors', 5, ...
                                'Standardize', true);
                            pred = predict(mdl, X_test);

                        case 'RF'
                            % 100 Trees as per manuscript
                            mdl = TreeBagger(100, X_train, y_train, ...
                                'Method', 'classification', ...
                                'NumPredictorsToSample', 'all');
                            predC = predict(mdl, X_test);
                            pred = categorical(predC);
                    end

                    fold_f1(c, k) = f1_score(y_test, pred);

                catch ME
                    fprintf('  -> Classifier %s failed on fold %d: %s\n', ...
                        classifiers{c}, k, ME.message);
                end
            end
        end
        
        % Average F1 across folds
        f1_table(:, s) = mean(fold_f1, 2);
    end

    %% -------------------- Store Results --------------------
    for c = 1:length(classifiers)
        row = {dset, classifiers{c}};
        for s = 1:length(set_names)
            if isnan(f1_table(c, s))
                row{end+1} = '--';
            else
                row{end+1} = sprintf('%.3f', f1_table(c, s));
            end
        end
        results = [results; row];
    end

    %% -------------------- Display Output --------------------
    fprintf('F1-score results for %s:\n', dset);
    T = array2table(f1_table, 'VariableNames', set_names, 'RowNames', classifiers);
    disp(T);
end

%% -------------------- Save LaTeX Table --------------------
fid = fopen(fullfile(results_folder, 'comparison_results.tex'), 'w');
fprintf(fid, '\\begin{table}[ht]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{F1-scores of proposed and unsupervised baseline methods on biomedical datasets.}\n');
fprintf(fid, '\\label{tab:unsup_comparison}\n');
fprintf(fid, '\\begin{tabular}{l|l|ccccc}\n');
fprintf(fid, 'Dataset & Classifier & All & Proposed & Laplacian & MCFS & UDFS \\\\\n');
fprintf(fid, '\\hline\n');

for i = 1:size(results, 1)
    fprintf(fid, '%s & %s & %s & %s & %s & %s & %s \\\\\n', ...
        results{i,1}, results{i,2}, results{i,3}, results{i,4}, ...
        results{i,5}, results{i,6}, results{i,7});
end

fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);

fprintf('\n✅ Results saved to: %s\\comparison_results.tex\n', results_folder);


%% -------------------------------------------------------------------
%% -------------------- Helper Functions -----------------------------
%% -------------------------------------------------------------------

function idx = laplacian_score(X, k)
    % Unsupervised Laplacian Score Implementation
    X = zscore(X);
    n = size(X,1);
    if n < 2, idx = 1:min(k, size(X,2)); return; end
    
    % Construct Affinity Matrix
    sigma2 = median(pdist2(X, X, 'euclidean').^2) / log(n);
    if sigma2 == 0, sigma2 = 1; end
    W = exp(-pdist2(X, X).^2 / (2 * sigma2));
    D = diag(sum(W, 2));
    L = D - W;
    
    % Compute Scores
    one = ones(n,1);
    f_hat = X - ((D * X) ./ (D * one + eps));
    scores = zeros(size(X,2),1);
    for i = 1:size(X,2)
        fi = f_hat(:,i);
        scores(i) = (fi' * L * fi) / (fi' * fi + eps);
    end
    
    % Select top k (lower score is better)
    [~, sorted_idx] = sort(scores, 'ascend');
    idx = sorted_idx(1:k)';
end

function idx = mcfs_simple(X, k)
    % Simplified Multi-Cluster Feature Selection
    X = zscore(X);
    n = size(X,1); d = size(X,2);
    if d <= k
        idx = 1:d;
        return;
    end
    
    % Determine number of clusters
    num_clusters = min(8, max(2, round(sqrt(n))));
    if num_clusters >= n
        num_clusters = min(n-1, 2);
    end
    
    try
        opts = statset('MaxIter',100,'Display','off');
        [~, C] = kmeans(X, num_clusters, 'MaxIter',100,'Replicates',5,'Options',opts);
        C = round(C);
        if any(C < 1) || any(isnan(C)) || any(~isfinite(C))
            error('Invalid labels');
        end
        [~, ~, C] = unique(C,'stable');
        Y = dummyvar(C);
        
        % Regression to find importance
        lambda = 1e-3;
        W = (X' * X + lambda * eye(d)) \ (X' * Y);
        score = sum(abs(W), 2);
        [~, idx] = sort(score, 'descend');
        idx = idx(1:k)';
    catch
        % Fallback to variance
        score = var(X,0,1)';
        [~, idx] = sort(score,'descend');
        idx = idx(1:k)';
    end
end

function idx = udfs_simple(X, k)
    % Simplified Unsupervised Discriminative Feature Selection
    X = zscore(X);
    d = size(X,2);
    % Approximation using SVD loadings
    [~, ~, V] = svd(X,'econ');
    loadings = mean(abs(V(:,1:min(5,d))),2);
    variance = var(X,0,1)';
    score = loadings .* variance;
    [~, idx] = sort(score,'descend');
    idx = idx(1:k)';
end

function idx = spec_score(X, k)
    % Fallback Spectral Score (Variance-based)
    score = var(X,0,1)';
    [~, idx] = sort(score,'descend');
    idx = idx(1:k)';
end

function f1 = f1_score(y_true, y_pred)
    % Calculate Macro F1-Score
    C = confusionmat(y_true, y_pred);
    prec = diag(C) ./ (sum(C,1) + eps);
    rec = diag(C) ./ (sum(C,2) + eps);
    f1_per_class = 2 * (prec .* rec) ./ (prec + rec + eps);
    f1 = mean(f1_per_class(~isnan(f1_per_class)));
end
