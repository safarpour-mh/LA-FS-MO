%% Unsupervised Feature Selection via Fuzzy Crowding Distance
clear; clc; close all;

% Folders
data_folder = 'C:\Users\Administrator\Desktop\matlab\dataset\';
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% List of datasets
datasets = {'crx','australian','heart','ionosphere','wpbc','wdbc','segment','zoo'};

n_datasets = length(datasets);
results = struct();

% Fixed parameters from the paper
K = 5;          % bins for entropy
eps_val = 1e-6; % epsilon for DBI inversion

for idx = 1:n_datasets
    dataset_name = datasets{idx};
    fprintf('Processing dataset: %s\n', dataset_name);
    
    % Load data
    filename = fullfile(data_folder, [dataset_name, 'Samples.mat']);
    if ~isfile(filename)
        warning('File not found: %s', filename);
        continue;
    end
    data = load(filename);
    if ~isfield(data, 'X')
        error('File %s does not contain variable "X"', filename);
    end
    X = double(data.X); % Ensure numeric
    
    % Handle NaNs
    if any(isnan(X(:)))
        warning('NaNs detected in %s. Replacing with column medians.', dataset_name);
        for j = 1:size(X, 2)
            col = X(:, j);
            col(isnan(col)) = median(col(~isnan(col)));
            X(:, j) = col;
        end
    end
    
    [n, d] = size(X);
    o1 = zeros(d, 1); % Entropy
    o2 = zeros(d, 1); % Inverted DBI
    
    % Evaluate features
    for j = 1:d
        x = X(:, j);
        if range(x) < eps
            o1(j) = 0; o2(j) = 0;
            continue;
        end
        
        % Entropy
        edges = linspace(min(x), max(x), K+1);
        [~,~,bin_idx] = histcounts(x, edges);
        pk = accumarray(bin_idx, 1)/n;
        pk = pk(pk>0);
        o1(j) = -sum(pk.*log2(pk+eps));
        
        % Inverted DBI
        try
            [idx_k, C] = kmeans(x, 2, 'Replicates', 5, 'MaxIter', 100, 'Display', 'off');
            sigma1 = std(x(idx_k==1)); sigma2 = std(x(idx_k==2));
            sigma1 = max(sigma1, 0); sigma2 = max(sigma2, 0);
            dist = abs(C(1)-C(2));
            if dist < eps
                dbi = Inf;
            else
                dbi = 0.5 * (sigma1 / dist + sigma2 / dist);
            end
            o2(j) = 1/(dbi + eps_val);
        catch
            o2(j) = 0;
        end
    end
    
    % Normalize
    o1 = (o1 - min(o1)) / (max(o1)-min(o1)+eps);
    o2 = (o2 - min(o2)) / (max(o2)-min(o2)+eps);
    
    % Fuzzy crowding distance
    CD_fuzzy = computeFuzzyCrowdingDistance(o1, o2);
    
    % Select features
    r = max(5, ceil(0.2*d));
    [~, sorted_idx] = sort(CD_fuzzy, 'descend');
    selected_features = sorted_idx(1:r);
    
    % Store results
    results.(dataset_name).dataset = dataset_name;
    results.(dataset_name).n = n;
    results.(dataset_name).d = d;
    results.(dataset_name).r = r;
    results.(dataset_name).entropy_scores = o1;
    results.(dataset_name).inv_dbi_scores = o2;
    results.(dataset_name).CD_fuzzy = CD_fuzzy;
    results.(dataset_name).selected_features = selected_features;
    results.(dataset_name).feature_ranks = rank(-CD_fuzzy);
    
    fprintf('Selected top %d/%d features for %s: %s\n', r, d, dataset_name, mat2str(selected_features'));
end

% Save results
save(fullfile(results_folder,'feature_selection_results.mat'), 'results');
fprintf('\nAll results saved to %s\n', fullfile(results_folder,'feature_selection_results.mat'));

%% Helper function
function CD = computeFuzzyCrowdingDistance(obj1, obj2)
    d = length(obj1);
    CD = zeros(d,1);
    [~, idx1] = sort(obj1,'descend'); [~, idx2] = sort(obj2,'descend');
    mu1_plus=zeros(d,1); mu1_minus=zeros(d,1); mu2_plus=zeros(d,1); mu2_minus=zeros(d,1);
    sigma1=(max(obj1)-min(obj1))/10+eps; sigma2=(max(obj2)-min(obj2))/10+eps;
    
    for i=1:d
        pos=find(idx1==i);
        if pos==1 || pos==d
            mu1_plus(i)=1; mu1_minus(i)=1;
        else
            delta_plus=obj1(idx1(pos))-obj1(idx1(pos+1));
            delta_minus=obj1(idx1(pos-1))-obj1(idx1(pos));
            mu1_plus(i)=1-exp(-(delta_plus/sigma1)^2);
            mu1_minus(i)=1-exp(-(delta_minus/sigma1)^2);
        end
        
        pos=find(idx2==i);
        if pos==1 || pos==d
            mu2_plus(i)=1; mu2_minus(i)=1;
        else
            delta_plus=obj2(idx2(pos))-obj2(idx2(pos+1));
            delta_minus=obj2(idx2(pos-1))-obj2(idx2(pos));
            mu2_plus(i)=1-exp(-(delta_plus/sigma2)^2);
            mu2_minus(i)=1-exp(-(delta_minus/sigma2)^2);
        end
    end
    CD=(mu1_plus+mu1_minus+mu2_plus+mu2_minus)/4;
end
