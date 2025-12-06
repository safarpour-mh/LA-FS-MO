%% =========================================================
% run_comparison_eight_datasets_safe_nonstratified.m
%% =========================================================

clear; clc; close all;

%% -------------------- Settings --------------------
data_folder = 'C:\Users\Administrator\Desktop\matlab\dataset\';
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

datasets = {'crx','australian','heart','ionosphere','wpbc','wdbc','segment','zoo'};
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
    Xfile = fullfile(data_folder, [datasets{d}, 'Samples.mat']);
    yfile = fullfile(data_folder, [datasets{d}, 'Label.mat']);
    
    X = double(load(Xfile).X);
    y = load(yfile).X; y = y(:);
    
    selected_features = selected_features_cell{d};
    X_selected = X(:, selected_features);
    
    %% Run classifiers
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
    
    % Mean ROC
    Y_all = [ safeInterp(scores_all.SVM, Xgrid), ...
              safeInterp(scores_all.KNN, Xgrid), ...
              safeInterp(scores_all.RF, Xgrid) ];
    Y_sel = [ safeInterp(scores_sel.SVM, Xgrid), ...
              safeInterp(scores_sel.KNN, Xgrid), ...
              safeInterp(scores_sel.RF, Xgrid) ];
          
    Y_all_mean = mean(Y_all, 2);  % اطمینان از اینکه ستون است
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


%% =========================================================
% Functions
%% =========================================================

function metrics = runClassifiers(X, y, k)
    cv = cvpartition(y,'KFold',k,'Stratify',false);  % non-stratified to remove warnings
    for i = 1:k
        Xtr = X(training(cv,i),:);  
        ytr = y(training(cv,i));
        Xte = X(test(cv,i),:);     
        yte = y(test(cv,i));

        % --- SVM (multi-class safe) ---
        if numel(unique(ytr)) > 2
            m = fitcecoc(Xtr,ytr,'Learners',templateSVM('KernelFunction','rbf','Standardize',true));
        else
            m = fitcsvm(Xtr,ytr,'KernelFunction','rbf','Standardize',true);
            m = fitPosterior(m);
        end
        yp = predict(m,Xte);
        SVM_metrics(i) = calc(yte, yp);

        % --- KNN ---
        m = fitcknn(Xtr,ytr,'NumNeighbors',5,'Standardize',true);
        yp = predict(m,Xte);
        KNN_metrics(i) = calc(yte, yp);

        % --- RF ---
        m = fitcensemble(Xtr,ytr,'Method','Bag');
        yp = predict(m,Xte);
        RF_metrics(i) = calc(yte, yp);
    end
    metrics.SVM = meanStruct(SVM_metrics);
    metrics.KNN = meanStruct(KNN_metrics);
    metrics.RF  = meanStruct(RF_metrics);
end

function out = calc(y, yp)
    out.Acc = mean(y==yp);
    out.Prec = sum(yp==1 & y==1)/(sum(yp==1)+eps);
    out.Rec = sum(yp==1 & y==1)/(sum(y==1)+eps);
    out.F1 = 2*out.Prec*out.Rec/(out.Prec+out.Rec+eps);
end

function m = meanStruct(s)
    f = fieldnames(s);
    for i=1:numel(f)
        m.(f{i}) = mean([s.(f{i})]);
    end
end

function Ynew = safeInterp(S, Xgrid)
    [Xroc,Yroc,~,~] = perfcurve(S.ytrue, S.score,1);
    [Xunique, idx] = unique(Xroc,'stable');
    Yunique = Yroc(idx);
    Ynew = interp1(Xunique,Yunique,Xgrid,'linear','extrap');
end

function Scores = getCVScores_for_ROC(X, y, k)
    cv = cvpartition(y,'KFold',k,'Stratify',false);  % non-stratified
    algos = {'SVM','KNN','RF'};
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
            s = double(yp==1); % ساده‌سازی score برای ROC دودویی (چندکلاسه: برای مقایسه می‌توان کلاس هدف را انتخاب کرد)
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
        m = fitcensemble(Xtr,ytr,'Method','Bag');
        [~,s] = predict(m,Xte);
        Scores.RF.ytrue = [Scores.RF.ytrue; yte];
        Scores.RF.score = [Scores.RF.score; s(:,2)];
    end
end
