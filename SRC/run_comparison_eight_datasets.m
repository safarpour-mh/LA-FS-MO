%% =========================================================
% run_comparison_eight_datasets.m
% Purpose: Reproduce classification results for all 8 datasets
%          Comparing Full vs. Selected Features.
%          Classifiers: SVM, KNN, Random Forest (5-fold CV).
%
% Requirements: MATLAB R2023a (Statistics and Machine Learning Toolbox)
% Usage: 
%   1. Place this script in the root folder.
%   2. Ensure a subfolder named 'dataset' exists containing .mat files.
%   3. Run the script. Results will be saved to /results folder.
%% =========================================================

clear; clc; close all;

%% -------------------- Reproducibility Setup --------------------
% Set random seed to ensure results are identical across runs
rng(42); 

%% -------------------- Settings --------------------
% Use relative path for portability
data_folder = 'dataset';
results_folder = 'results';

if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

datasets = {'crx','australian','heart','ionosphere','wpbc','wdbc','segment','zoo'};

% Selected feature indices based on the proposed method (Table 4 in manuscript)
selected_features_cell = {...
    [11, 9, 5, 3, 13], ...
    [13, 1, 14, 5, 4], ...
    [2, 5, 7, 1, 6], ...
    [2, 1, 3, 24, 7, 5, 4], ...
    [1, 17, 2, 27, 34, 18, 3], ...
    [17, 21, 28, 14, 11, 27], ...
    [3, 19, 4, 1, 5], ...
    [13, 1, 11, 16, 7]};

classifiers = {'SVM','KNN','RF'};
kfold = 5;
Xgrid = linspace(0,1,400)';

%% -------------------- Combined ROC Figure --------------------
figure('Units','centimeters','Position',[2 2 24 20]); 
tiledlayout(2,4, 'TileSpacing','compact', 'Padding','compact');

%% -------------------- Loop over datasets --------------------
for d = 1:length(datasets)
    
    %% Load Data
    % Expecting files named e.g., 'crxSamples.mat' and 'crxLabel.mat'
    Xfile = fullfile(data_folder, [datasets{d}, 'Samples.mat']);
    yfile = fullfile(data_folder, [datasets{d}, 'Label.mat']);
    
    % Check existence to prevent runtime errors
    if ~exist(Xfile, 'file') || ~exist(yfile, 'file')
        warning('Data files for %s not found. Skipping...', datasets{d});
        continue;
    end
    
    X = double(load(Xfile).X);
    y = load(yfile).X; y = y(:);
    
    selected_features = selected_features_cell{d};
    X_selected = X(:, selected_features);
    
    %% Run classifiers
    fprintf('Processing %s...\n', datasets{d});
    metrics_all = runClassifiers(X, y, kfold);
    metrics_selected = runClassifiers(X_selected, y, kfold);
    
    %% Save Results Table
    FeatureType = repmat({'All';'All';'All';'Selected';'Selected';'Selected'},1,1);
    ClassCol = [classifiers'; classifiers'];
    
    Acc  = [metrics_all.SVM.Acc, metrics_all.KNN.Acc, metrics_all.RF.Acc, ...
            metrics_selected.SVM.Acc, metrics_selected.KNN.Acc, metrics_selected.RF.Acc]';
    Prec = [metrics_all.SVM.Prec, metrics_all.KNN.Prec, metrics_all.RF.Prec, ...
            metrics_selected.SVM.Prec, metrics_selected.KNN.Prec, metrics_selected.RF.Prec]';
    Rec  = [metrics_all.SVM.Rec, metrics_all.KNN.Rec, metrics_all.RF.Rec, ...
            metrics_selected.SVM.Rec, metrics_selected.KNN.Rec, metrics_selected.RF.Rec]';
    F1s  = [metrics_all.SVM.F1, metrics_all.KNN.F1, metrics_all.RF.F1, ...
            metrics_selected.SVM.F1, metrics_selected.KNN.F1, metrics_selected.RF.F1]';
        
    ResultsTable = table(FeatureType, ClassCol, Acc, Prec, Rec, F1s);
    writetable(ResultsTable, fullfile(results_folder, [datasets{d}, '_Results.xlsx']));
    
    %% -------------------- ROC --------------------
    scores_all = getCVScores_for_ROC(X, y, kfold);
    scores_sel = getCVScores_for_ROC(X_selected, y, kfold);
    
    % Mean ROC across 3 classifiers
    Y_all = [ safeInterp(scores_all.SVM, Xgrid), ...
              safeInterp(scores_all.KNN, Xgrid), ...
              safeInterp(scores_all.RF, Xgrid) ];
    Y_sel = [ safeInterp(scores_sel.SVM, Xgrid), ...
              safeInterp(scores_sel.KNN, Xgrid), ...
              safeInterp(scores_sel.RF, Xgrid) ];
          
    Y_all_mean = mean(Y_all, 2);  % Ensure column vector orientation
    Y_sel_mean = mean(Y_sel, 2);
    
    % Plot
    nexttile; hold on;
    h(1) = plot(Xgrid, Y_all_mean, 'b-', 'LineWidth',2);
    h(2) = plot(Xgrid, Y_sel_mean, 'r--', 'LineWidth',2);
    plot([0 1],[0 1],'k:','LineWidth',1.5,'HandleVisibility','off');
    xlabel('FPR'); ylabel('TPR'); title(datasets{d},'Interpreter','none');
    legend(h, {'All Features','Selected Features'}, 'Location','SouthEast','FontSize',8);
    grid on; box on;
    
end

%% -------------------- Save combined ROC figure --------------------
set(gcf,'PaperPositionMode','auto');
print(fullfile(results_folder,'ROC_all_datasets.pdf'),'-dpdf','-bestfit');
saveas(gcf, fullfile(results_folder,'ROC_all_datasets.png'));

fprintf('All results saved to /%s folder.\n', results_folder);

%% =========================================================
% Helper Functions
%% =========================================================

function metrics = runClassifiers(X, y, k)
    % Stratified K-Fold Cross-Validation (matches manuscript methodology)
    cv = cvpartition(y,'KFold',k,'Stratify',true);
    
    SVM_metrics = struct([]);
    KNN_metrics = struct([]);
    RF_metrics = struct([]);

    for i = 1:k
        Xtr = X(training(cv,i),:);  
        ytr = y(training(cv,i));
        Xte = X(test(cv,i),:);     
        yte = y(test(cv,i));

        % --- SVM (RBF Kernel) ---
        % Note: Full hyperparameter tuning was performed in the study.
        % This script uses standard parameters for reproducibility demonstration.
        if numel(unique(ytr)) > 2
            m = fitcecoc(Xtr,ytr,'Learners',templateSVM('KernelFunction','rbf','Standardize',true));
        else
            m = fitcsvm(Xtr,ytr,'KernelFunction','rbf','Standardize',true);
            m = fitPosterior(m);
        end
        yp = predict(m,Xte);
        SVM_metrics(i) = calc(yte, yp);

        % --- KNN (k=5) ---
        m = fitcknn(Xtr,ytr,'NumNeighbors',5,'Standardize',true);
        yp = predict(m,Xte);
        KNN_metrics(i) = calc(yte, yp);

        % --- Random Forest (100 Trees) ---
        % Explicitly setting NumLearn=100 to match manuscript
        m = fitcensemble(Xtr,ytr,'Method','Bag', 'NumLearn', 100);
        yp = predict(m,Xte);
        RF_metrics(i) = calc(yte, yp);
    end
    
    % Handle cases where metrics might be empty due to errors
    if ~isempty(SVM_metrics)
        metrics.SVM = meanStruct(SVM_metrics);
        metrics.KNN = meanStruct(KNN_metrics);
        metrics.RF  = meanStruct(RF_metrics);
    else
        error('Classification failed for dataset.');
    end
end

function out = calc(y, yp)
    % Calculate Performance Metrics
    % Note: For multi-class datasets, this computes metrics for Class 1 vs Rest
    out.Acc = mean(y==yp);
    out.Prec = sum(yp==1 & y==1)/(sum(yp==1)+eps);
    out.Rec = sum(yp==1 & y==1)/(sum(y==1)+eps);
    out.F1 = 2*out.Prec*out.Rec/(out.Prec+out.Rec+eps);
end

function m = meanStruct(s)
    % Average metrics across folds
    f = fieldnames(s);
    for i=1:numel(f)
        m.(f{i}) = mean([s.(f{i})]);
    end
end

function Ynew = safeInterp(S, Xgrid)
    % Compute ROC curve and interpolate to common grid
    [Xroc,Yroc,~,~] = perfcurve(S.ytrue, S.score, 1);
    [Xunique, idx] = unique(Xroc,'stable');
    Yunique = Yroc(idx);
    Ynew = interp1(Xunique,Yunique,Xgrid,'linear','extrap');
end

function Scores = getCVScores_for_ROC(X, y, k)
    % Stratified partition for ROC score collection
    cv = cvpartition(y,'KFold',k,'Stratify',true);
    algos = {'SVM','KNN','RF'};
    
    % Initialize structure
    for j = 1:numel(algos)
        Scores.(algos{j}).ytrue = [];
        Scores.(algos{j}).score = [];
    end
    
    for i = 1:cv.NumTestSets
        Xtr = X(training(cv,i),:);  
        ytr = y(training(cv,i));
        Xte = X(test(cv,i),:);     
        yte = y(test(cv,i));

        % --- SVM ---
        if numel(unique(ytr)) > 2
            m = fitcecoc(Xtr,ytr,'Learners',templateSVM('KernelFunction','rbf','Standardize',true));
            yp = predict(m,Xte);
            % Simplify score for binary ROC (Multi-class: target class selected for comparison)
            s = double(yp==1); 
        else
            m = fitcsvm(Xtr,ytr,'KernelFunction','rbf','Standardize',true);
            m = fitPosterior(m);
            [~,s] = predict(m,Xte);
            s = s(:,2);
        end
        Scores.SVM.ytrue = [Scores.SVM.ytrue; yte];
        Scores.SVM.score = [Scores.SVM.score; s];

        % --- KNN ---
        m = fitcknn(Xtr,ytr,'NumNeighbors',5,'Standardize',true);
        [~,s] = predict(m,Xte);
        Scores.KNN.ytrue = [Scores.KNN.ytrue; yte];
        Scores.KNN.score = [Scores.KNN.score; s(:,2)];

        % --- RF ---
        m = fitcensemble(Xtr,ytr,'Method','Bag', 'NumLearn', 100);
        [~,s] = predict(m,Xte);
        Scores.RF.ytrue = [Scores.RF.ytrue; yte];
        Scores.RF.score = [Scores.RF.score; s(:,2)];
    end
end
