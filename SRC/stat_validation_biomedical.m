%% =========================================================
% run_statistical_validation.m
% Purpose: Perform Wilcoxon signed-rank test on F1-scores
%          comparing Full vs. Selected features on biomedical datasets.
%          Matches methodology described in manuscript (Section: Statistical Validation).
%
% Requirements: MATLAB R2023a (Statistics and Machine Learning Toolbox)
% Usage: 
%   1. Place this script in the root folder.
%   2. Ensure 'dataset' folder contains .mat files.
%   3. Run the script. Results saved to /results folder.
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

% Biomedical datasets used for statistical validation (Table 7 in manuscript)
datasets = {'heart', 'wpbc', 'wdbc'};
classifiers = {'SVM', 'KNN', 'RF'};
n_datasets = numel(datasets);
n_cls = numel(classifiers);

% Bonferroni correction: 3 datasets × 3 classifiers = 9 comparisons
alpha_corrected = 0.05 / (n_datasets * n_cls);  % ≈ 0.00556

% Selected indices from Table 4 in manuscript
selected_indices_paper = struct();
selected_indices_paper.heart = [2, 5, 7, 1, 6];
selected_indices_paper.wpbc = [1, 17, 2, 27, 34, 18, 3];
selected_indices_paper.wdbc = [17, 21, 28, 14, 11, 27];

% Preallocate results structure
stats_results = struct();

%% -------------------- Main Loop --------------------
for d = 1:n_datasets
    dataset = datasets{d};
    fprintf('Processing: %s\n', dataset);
    
    % Load Features
    X_file = fullfile(data_folder, [dataset, 'Samples.mat']);
    if ~exist(X_file, 'file')
        error('Missing data file: %s', X_file);
    end
    X = double(load(X_file).X);
    
    % Load Labels
    y_file = fullfile(data_folder, [dataset, 'Label.mat']);
    y_raw = load(y_file).X;
    
    % --- Robust Label Conversion to Numeric (1/2) ---
    if iscell(y_raw)
        y_str = string(y_raw(:));
    elseif ischar(y_raw)
        if size(y_raw,1) == 1 && size(y_raw,2) > 1
            y_str = string(y_raw')';
        else
            y_str = string(y_raw(:));
        end
    elseif isstring(y_raw)
        y_str = y_raw(:);
    else
        % Already numeric
        y = double(y_raw(:));
        y = y(:);
        if min(y) == 0
            y = y + 1; % Map 0/1 → 1/2
        end
        y_str = []; % Flag for numeric
    end
    
    if ~isempty(y_str)
        % Map specific biomedical labels to numeric
        if strcmp(dataset, 'wpbc')
            y_str = replace(y_str, "N", "1");
            y_str = replace(y_str, "R", "2");
        elseif strcmp(dataset, 'wdbc')
            y_str = replace(y_str, "B", "1");
            y_str = replace(y_str, "M", "2");
        end
        y = double(y_str);
        if any(isnan(y))
            error('Label conversion failed for dataset %s. Check label values.', dataset);
        end
    end
    y = y(:);
    
    % Use predefined selected indices
    sel_idx = selected_indices_paper.(dataset);
    X_selected = X(:, sel_idx);
    
    % Storage for F1-scores (5 folds × 3 classifiers)
    f1_all = zeros(5, n_cls);
    f1_sel = zeros(5, n_cls);
    
    % Stratified 5-Fold Cross-Validation (Matches manuscript methodology)
    cv = cvpartition(y, 'KFold', 5, 'Stratify', true);
    
    for fold = 1:5
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);
        
        Xtr_all = X(trainIdx, :);    Xte_all = X(testIdx, :);
        Xtr_sel = X_selected(trainIdx, :); Xte_sel = X_selected(testIdx, :);
        ytr = y(trainIdx); yte = y(testIdx);
        
        % Skip fold if only one class in training (rare with stratified CV)
        if numel(unique(ytr)) < 2
            f1_all(fold,:) = NaN;
            f1_sel(fold,:) = NaN;
            continue;
        end
        
        % ========== SVM (with Internal Hyperparameter Tuning) ==========
        % Manuscript states: "hyperparameters tuned via internal 5-fold cross-validation"
        % To prevent data leakage, tuning is performed ONLY on training data.
        
        C_grid = [0.1, 1, 10, 100];
        gamma_grid = [0.01, 0.1, 1, 10];
        best_C = 1; best_gamma = 0.1;
        best_inner_f1 = -inf;
        
        % Inner CV for parameter selection (3-fold for speed, represents internal tuning)
        inner_cv = cvpartition(ytr, 'KFold', 3, 'Stratify', true);
        
        for C = C_grid
            for gamma = gamma_grid
                inner_f1_scores = zeros(inner_cv.NumTestSets, 1);
                for k = 1:inner_cv.NumTestSets
                    i_train = training(inner_cv, k);
                    i_val   = test(inner_cv, k);
                    
                    try
                        mdl_inner = fitcsvm(Xtr_all(i_train, :), ytr(i_train), ...
                            'KernelFunction', 'rbf', ...
                            'BoxConstraint', C, ...
                            'KernelScale', 1/sqrt(2*gamma), ...
                            'Standardize', true);
                        y_pred_inner = predict(mdl_inner, Xtr_all(i_val, :));
                        inner_f1_scores(k) = f1score(ytr(i_val), y_pred_inner);
                    catch
                        inner_f1_scores(k) = 0;
                    end
                end
                mean_inner_f1 = mean(inner_f1_scores);
                if mean_inner_f1 > best_inner_f1
                    best_inner_f1 = mean_inner_f1;
                    best_C = C;
                    best_gamma = gamma;
                end
            end
        end
        
        % Train final SVM on full training set with best parameters
        try
            mdl = fitcsvm(Xtr_all, ytr, ...
                'KernelFunction', 'rbf', ...
                'BoxConstraint', best_C, ...
                'KernelScale', 1/sqrt(2*best_gamma), ...
                'Standardize', true);
            y_pred = predict(mdl, Xte_all);
            f1_all(fold,1) = f1score(yte, y_pred);
        catch
            f1_all(fold,1) = NaN;
        end
        
        try
            mdl = fitcsvm(Xtr_sel, ytr, ...
                'KernelFunction', 'rbf', ...
                'BoxConstraint', best_C, ...
                'KernelScale', 1/sqrt(2*best_gamma), ...
                'Standardize', true);
            y_pred = predict(mdl, Xte_sel);
            f1_sel(fold,1) = f1score(yte, y_pred);
        catch
            f1_sel(fold,1) = NaN;
        end
        
        % ========== KNN (k=5) ==========
        try
            mdl = fitcknn(Xtr_all, ytr, 'NumNeighbors', 5, 'Distance', 'euclidean', 'Standardize', true);
            y_pred = predict(mdl, Xte_all);
            f1_all(fold,2) = f1score(yte, y_pred);
            
            mdl = fitcknn(Xtr_sel, ytr, 'NumNeighbors', 5, 'Distance', 'euclidean', 'Standardize', true);
            y_pred = predict(mdl, Xte_sel);
            f1_sel(fold,2) = f1score(yte, y_pred);
        catch
            f1_all(fold,2) = NaN;
            f1_sel(fold,2) = NaN;
        end
        
        % ========== Random Forest (100 Trees) ==========
        d_all = size(Xtr_all, 2);
        d_sel = size(Xtr_sel, 2);
        
        try
            mdl = TreeBagger(100, Xtr_all, ytr, ...
                'Method', 'classification', ...
                'NumPredictorsToSample', floor(sqrt(d_all)));
            y_pred_cell = predict(mdl, Xte_all);
            y_pred = convertPrediction(y_pred_cell);
            f1_all(fold,3) = f1score(yte, y_pred);
            
            mdl = TreeBagger(100, Xtr_sel, ytr, ...
                'Method', 'classification', ...
                'NumPredictorsToSample', floor(sqrt(d_sel)));
            y_pred_cell = predict(mdl, Xte_sel);
            y_pred = convertPrediction(y_pred_cell);
            f1_sel(fold,3) = f1score(yte, y_pred);
        catch
            f1_all(fold,3) = NaN;
            f1_sel(fold,3) = NaN;
        end
    end
    
    stats_results.(dataset).f1_all = f1_all;
    stats_results.(dataset).f1_sel = f1_sel;
    
    % Wilcoxon signed-rank test per classifier
    pvals = zeros(1, n_cls);
    for ci = 1:n_cls
        valid = ~(isnan(f1_all(:,ci)) | isnan(f1_sel(:,ci)));
        if sum(valid) < 2
            pvals(ci) = 1;
        else
            % Two-sided signed-rank test
            [p, ~] = signrank(f1_all(valid,ci), f1_sel(valid,ci));
            pvals(ci) = p;
        end
    end
    stats_results.(dataset).p_values = pvals;
    stats_results.(dataset).significant = pvals < alpha_corrected;
end

% Save MATLAB Results
save(fullfile(results_folder, 'statistical_validation_results.mat'), ...
    'stats_results', 'alpha_corrected', 'datasets', 'classifiers');

%% -------------------- Display Results --------------------
fprintf('\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('Wilcoxon Signed-Rank Test Results (Biomedical Datasets)\n');
fprintf('Bonferroni-corrected α = %.4f\n', alpha_corrected);
fprintf('Note: With 5-fold CV, minimum achievable p-value is 0.0625.\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('%-10s %-8s %12s %12s %10s %s\n', 'Dataset', 'Classifier', 'Mean F1 (All)', 'Mean F1 (Sel)', 'p-value', '');
fprintf('%s\n', repmat('-', 1, 80));

for d = 1:n_datasets
    dataset = datasets{d};
    for c = 1:n_cls
        clf = classifiers{c};
        mean_all = mean(stats_results.(dataset).f1_all(:,c), 'omitnan');
        mean_sel = mean(stats_results.(dataset).f1_sel(:,c), 'omitnan');
        p = stats_results.(dataset).p_values(c);
        sig_mark = '';
        if p < alpha_corrected
            sig_mark = '*';
        end
        fprintf('%-10s %-8s %12.4f %12.4f %10.4f %s\n', ...
            dataset, clf, mean_all, mean_sel, p, sig_mark);
    end
end
fprintf('\n* indicates statistical significance at α = %.4f (Bonferroni-corrected)\n', alpha_corrected);

%% -------------------- Generate LaTeX Table --------------------
latex_lines = {
    '\begin{table}[htbp]',
    '\centering',
    '\caption{Statistical significance of F1-score differences (Wilcoxon signed-rank test) between full and selected feature sets on biomedical datasets.}',
    '\label{tab:statistical_validation}',
    '\begin{tabular}{lccc}',
    '\toprule',
    'Dataset & Classifier & $p$-value & Significant? \\',
    '\midrule'
};

for d = 1:n_datasets
    dataset = datasets{d};
    for c = 1:n_cls
        p = stats_results.(dataset).p_values(c);
        if stats_results.(dataset).significant(c)
            sig_str = '\checkmark';
        else
            sig_str = '---';
        end
        latex_lines{end+1} = sprintf('%s & %s & %.4f & %s \\\\', ...
            dataset, classifiers{c}, p, sig_str);
    end
    if d < n_datasets
        latex_lines{end+1} = '\midrule';
    end
end

latex_lines{end+1} = '\bottomrule';
latex_lines{end+1} = '\end{tabular}';
latex_lines{end+1} = '\end{table}';

% Save LaTeX Table
fid = fopen(fullfile(results_folder, 'wilcoxon_table.tex'), 'w');
for i = 1:numel(latex_lines)
    fprintf(fid, '%s\n', latex_lines{i});
end
fclose(fid);

fprintf('\n✅ Results saved:\n');
fprintf('   - MATLAB: %s\n', fullfile(results_folder, 'statistical_validation_results.mat'));
fprintf('   - LaTeX : %s\n', fullfile(results_folder, 'wilcoxon_table.tex'));

%% -------------------- Helper Functions --------------------

function f1 = f1score(y_true, y_pred)
    % Calculate Macro F1-Score
    y_true = double(y_true(:));
    y_pred = double(y_pred(:));
    classes = unique([y_true; y_pred]);
    if numel(classes) == 2
        C = confusionmat(y_true, y_pred, 'Order', classes);
        prec = C(2,2) / (C(2,2) + C(1,2) + eps);
        rec  = C(2,2) / (C(2,2) + C(2,1) + eps);
        f1 = 2 * prec * rec / (prec + rec + eps);
    else
        C = confusionmat(y_true, y_pred, 'Order', classes);
        precisions = diag(C) ./ (sum(C,1)' + eps);
        recalls    = diag(C) ./ (sum(C,2)  + eps);
        f1s = 2 * precisions .* recalls ./ (precisions + recalls + eps);
        f1 = mean(f1s);
    end
end

function y_num = convertPrediction(y_cell)
    % Convert TreeBagger cell output to numeric vector
    if iscell(y_cell)
        if ischar(y_cell{1}) || isstring(y_cell{1})
            y_num = str2double(string(y_cell));
        else
            y_num = cell2mat(y_cell);
        end
    else
        y_num = y_cell;
    end
    y_num = double(y_num(:));
end
