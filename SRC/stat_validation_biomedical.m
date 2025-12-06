
clc;
clear;
close all;

%% -------------------- Settings --------------------
data_folder = 'C:\Users\Administrator\Desktop\matlab\dataset\';
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

datasets = {'heart','wpbc','wdbc'};
classifiers = {'SVM', 'KNN', 'RF'};
n_datasets = numel(datasets);
n_cls = numel(classifiers);

% Bonferroni correction: 3 datasets × 3 classifiers = 9 comparisons
alpha_corrected = 0.05 / (n_datasets * n_cls);  % ≈ 0.00556

% Selected indices from Table 4 in your paper
selected_indices_paper = struct();
selected_indices_paper.heart = [2, 5, 7, 1, 6];
selected_indices_paper.wpbc = [1, 17, 2, 27, 34, 18, 3];
selected_indices_paper.wdbc = [17, 21, 28, 14, 11, 27];

% Preallocate
stats_results = struct();

%% -------------------- Main Loop --------------------
for d = 1:n_datasets
    dataset = datasets{d};
    fprintf('Processing: %s\n', dataset);
    
    % Load features (variable 'X')
    X = double(load(fullfile(data_folder, [dataset, 'Samples.mat'])).X);
    
    % Load raw labels (variable 'X' in Label.mat — may be char/cell/numeric)
    y_raw = load(fullfile(data_folder, [dataset, 'Label.mat'])).X;
    
    % --- Convert labels to numeric (1/2) robustly ---
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
        % Skip string conversion
        y_str = []; % Flag for numeric
    end
    
    if ~isempty(y_str)
        % Map string labels to numeric for biomedical datasets
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
    
    % 5-fold CV
    cv = cvpartition(y, 'KFold', 5);
    
    for fold = 1:5
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);
        
        Xtr_all = X(trainIdx, :);    Xte_all = X(testIdx, :);
        Xtr_sel = X_selected(trainIdx, :); Xte_sel = X_selected(testIdx, :);
        ytr = y(trainIdx); yte = y(testIdx);
        
        % Skip fold if only one class in training (should not happen with stratified CV)
        if numel(unique(ytr)) < 2
            f1_all(fold,:) = NaN;
            f1_sel(fold,:) = NaN;
            continue;
        end
        
        % ========== SVM ==========
        C_grid = [0.1, 1, 10, 100];
        gamma_grid = [0.01, 0.1, 1, 10];
        
        best_f1 = -inf;
        for C = C_grid
            for gamma = gamma_grid
                try
                    mdl = fitcsvm(Xtr_all, ytr, ...
                        'KernelFunction', 'rbf', ...
                        'BoxConstraint', C, ...
                        'KernelScale', 1/sqrt(2*gamma), ...
                        'Standardize', false, ...
                        'ClassNames', unique(ytr));
                    y_pred = predict(mdl, Xte_all);
                    f1_temp = f1score(yte, y_pred);
                    if f1_temp > best_f1, best_f1 = f1_temp; end
                catch
                    % Ignore numerical errors
                end
            end
        end
        f1_all(fold,1) = best_f1;
        
        best_f1 = -inf;
        for C = C_grid
            for gamma = gamma_grid
                try
                    mdl = fitcsvm(Xtr_sel, ytr, ...
                        'KernelFunction', 'rbf', ...
                        'BoxConstraint', C, ...
                        'KernelScale', 1/sqrt(2*gamma), ...
                        'Standardize', false);
                    y_pred = predict(mdl, Xte_sel);
                    f1_temp = f1score(yte, y_pred);
                    if f1_temp > best_f1, best_f1 = f1_temp; end
                catch
                end
            end
        end
        f1_sel(fold,1) = best_f1;
        
        % ========== KNN ==========
        mdl = fitcknn(Xtr_all, ytr, 'NumNeighbors', 5, 'Distance', 'euclidean');
        y_pred = predict(mdl, Xte_all);
        f1_all(fold,2) = f1score(yte, y_pred);
        
        mdl = fitcknn(Xtr_sel, ytr, 'NumNeighbors', 5, 'Distance', 'euclidean');
        y_pred = predict(mdl, Xte_sel);
        f1_sel(fold,2) = f1score(yte, y_pred);
        
        % ========== Random Forest ==========
        d_all = size(Xtr_all, 2);
        d_sel = size(Xtr_sel, 2);
        
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
    end
    
    stats_results.(dataset).f1_all = f1_all;
    stats_results.(dataset).f1_sel = f1_sel;
    
    % Wilcoxon test per classifier
    pvals = zeros(1, n_cls);
    for ci = 1:n_cls
        valid = ~(isnan(f1_all(:,ci)) | isnan(f1_sel(:,ci)));
        if sum(valid) < 2
            pvals(ci) = 1;
        else
            [p, ~] = signrank(f1_all(valid,ci), f1_sel(valid,ci));
            pvals(ci) = p;
        end
    end
    stats_results.(dataset).p_values = pvals;
    stats_results.(dataset).significant = pvals < alpha_corrected;
end

% Save
save(fullfile(results_folder, 'statistical_validation_biomedical_results.mat'), ...
    'stats_results', 'alpha_corrected', 'datasets', 'classifiers');

%% -------------------- Display Results --------------------

fprintf('\n');
fprintf('%s\n', repmat('=', 1, 80));
fprintf('Wilcoxon Signed-Rank Test Results (Biomedical Datasets)\n');
fprintf('Bonferroni-corrected α = %.4f\n', alpha_corrected);
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

% Save LaTeX
fid = fopen(fullfile(results_folder, 'wilcoxon_table.tex'), 'w');
for i = 1:numel(latex_lines)
    fprintf(fid, '%s\n', latex_lines{i});
end
fclose(fid);

fprintf('\n✅ Results saved:\n');
fprintf('   - MATLAB: %s\n', fullfile(results_folder, 'statistical_validation_biomedical_results.mat'));
fprintf('   - LaTeX : %s\n', fullfile(results_folder, 'wilcoxon_table.tex'));

%% -------------------- Helper Functions --------------------

function f1 = f1score(y_true, y_pred)
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