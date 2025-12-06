clc;
clear;
close all;

%% -------------------- Settings --------------------
data_folder = 'C:\Users\Administrator\Desktop\matlab\dataset';
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

datasets = {'heart', 'wpbc', 'wdbc'};

% Selected feature indices based on your paper
selected_indices_paper = struct();
selected_indices_paper.heart = [2, 5, 7, 1, 6];
selected_indices_paper.wpbc = [1, 17, 2, 27, 34, 18, 3];
selected_indices_paper.wdbc = [17, 21, 28, 14, 11, 27];

classifiers = {'SVM', 'KNN', 'RF'};
results = {};

%% -------------------- Main Loop --------------------
for dset = datasets
    fprintf('=============================================\n');
    fprintf('Processing dataset: %s\n', dset{1});
    fprintf('=============================================\n');

    % Load data
    X_file = fullfile(data_folder, [dset{1}, 'Samples.mat']);
    y_file = fullfile(data_folder, [dset{1}, 'Label.mat']);

    if ~exist(X_file, 'file') || ~exist(y_file, 'file')
        error('Missing .mat files for dataset: %s', dset{1});
    end

    X_all = load(X_file).X;
    y_vec = load(y_file).X;
    y = categorical(y_vec(:));

    r = length(selected_indices_paper.(dset{1}));

    % Proposed
    X_proposed = X_all(:, selected_indices_paper.(dset{1}));

    % LAP
    idx_lap = laplacian_score(X_all, r);
    X_lap = X_all(:, idx_lap);

    % MCFS
    idx_mcfs = mcfs_simple(X_all, r);
    X_mcfs = X_all(:, idx_mcfs);

    % UDFS
    try
        idx_udfs = udfs_simple(X_all, r);
        X_udfs = X_all(:, idx_udfs);
    catch
        fprintf('  -> UDFS failed, using SPEC.\n');
        idx_udfs = spec_score(X_all, r);
        X_udfs = X_all(:, idx_udfs);
    end

    % Evaluate
    X_sets = {X_all, X_proposed, X_lap, X_mcfs, X_udfs};
    set_names = {'All', 'Proposed', 'Laplacian', 'MCFS', 'UDFS'};
    f1_table = NaN(length(classifiers), length(set_names));

    for s = 1:length(X_sets)
        X = X_sets{s};

        % Remove zero-variance features
        stds = std(X, 0, 1);
        valid_cols = stds > 0;
        if ~any(valid_cols)
            continue;
        else
            X = X(:, valid_cols);
        end

        X_z = zscore(X);

        for c = 1:length(classifiers)
            try
                switch classifiers{c}
                    case 'SVM'
                        mdl = fitcsvm(X_z, y, 'KernelFunction', 'rbf', 'BoxConstraint', 1);

                    case 'KNN'
                        mdl = fitcknn(X_z, y, 'NumNeighbors', 5);

                    case 'RF'
                        mdl = TreeBagger(100, X_z, y, 'Method', 'classification');
                end

                if strcmp(classifiers{c}, 'RF')
                    predC = predict(mdl, X_z);
                    pred = categorical(predC);
                else
                    pred = predict(mdl, X_z);
                end

                f1 = f1_score(y, pred);
                f1_table(c, s) = f1;

            catch ME
                fprintf('  -> Classifier %s failed on %s (%s): %s\n', ...
                    classifiers{c}, dset{1}, set_names{s}, ME.message);
            end
        end
    end

    %% -------------------- Store Results --------------------
    for c = 1:length(classifiers)
        row = {dset{1}, classifiers{c}};
        for s = 1:length(set_names)
            if isnan(f1_table(c, s))
                row{end+1} = '--';
            else
                row{end+1} = sprintf('%.3f', f1_table(c, s));
            end
        end

        results = [results; row];
    end

    %% -------------------- DISPLAY OUTPUT --------------------
    fprintf('F1-score results for %s:\n', dset{1});
    T = array2table(f1_table, 'VariableNames', set_names, 'RowNames', classifiers);
    disp(T)
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
    X = zscore(X);
    n = size(X,1);
    if n < 2, idx = 1:min(k, size(X,2)); return; end
    sigma2 = median(pdist2(X, X, 'euclidean').^2) / log(n);
    if sigma2 == 0, sigma2 = 1; end
    W = exp(-pdist2(X, X).^2 / (2 * sigma2));
    D = diag(sum(W, 2));
    L = D - W;
    one = ones(n,1);
    f_hat = X - ((D * X) ./ (D * one + eps));
    scores = zeros(size(X,2),1);
    for i = 1:size(X,2)
        fi = f_hat(:,i);
        scores(i) = (fi' * L * fi) / (fi' * fi + eps);
    end
    [~, sorted_idx] = sort(scores, 'ascend');
    idx = sorted_idx(1:k)';
end

function idx = mcfs_simple(X, k)
    X = zscore(X);
    n = size(X,1); d = size(X,2);
    if d <= k
        idx = 1:d;
        return;
    end
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
        lambda = 1e-3;
        W = (X' * X + lambda * eye(d)) \ (X' * Y);
        score = sum(abs(W), 2);
        [~, idx] = sort(score, 'descend');
        idx = idx(1:k)';
    catch
        score = var(X,0,1)';
        [~, idx] = sort(score,'descend');
        idx = idx(1:k)';
    end
end

function idx = udfs_simple(X, k)
    X = zscore(X);
    d = size(X,2);
    [~, ~, V] = svd(X,'econ');
    loadings = mean(abs(V(:,1:min(5,d))),2);
    variance = var(X,0,1)';
    score = loadings .* variance;
    [~, idx] = sort(score,'descend');
    idx = idx(1:k)';
end

function idx = spec_score(X, k)
    score = var(X,0,1)';
    [~, idx] = sort(score,'descend');
    idx = idx(1:k)';
end

function f1 = f1_score(y_true, y_pred)
    C = confusionmat(y_true, y_pred);
    prec = diag(C) ./ (sum(C,1) + eps);
    rec = diag(C) ./ (sum(C,2) + eps);
    f1_per_class = 2 * (prec .* rec) ./ (prec + rec + eps);
    f1 = mean(f1_per_class(~isnan(f1_per_class)));
end
