%% =========================================================
% run_comparison_full.m
% - 5-fold CV
% - Metrics: Accuracy, Precision, Recall, F1
% - Classifiers: SVM, KNN, Random Forest
% - Compare All vs Selected Features
% - Mean ROC plot (SVM+KNN+RF)
% - Export Excel + PDF + PNG
%% =========================================================

clear; clc; close all;

%% -------------------- Load Data --------------------
X = load('C:\Users\Administrator\Desktop\New folder (2)\dataset\wdbcSamples.mat').X;
y = load('C:\Users\Administrator\Desktop\New folder (2)\dataset\wdbcLabel.mat').X;
y = y(:);
X = double(X);

%% -------------------- Selected Features --------------------
selected_features = [17 21 28 14 11 27];
X_selected = X(:, selected_features);

%% -------------------- Run Classifiers --------------------
metrics_all = runClassifiers(X, y);
metrics_selected = runClassifiers(X_selected, y);

%% -------------------- Create Results Table --------------------
Classifiers = {'SVM'; 'KNN'; 'RF'};
FeatureType = {'All'; 'All'; 'All'; 'Selected'; 'Selected'; 'Selected'};

Acc  = [metrics_all.SVM.Acc, metrics_all.KNN.Acc, metrics_all.RF.Acc, ...
        metrics_selected.SVM.Acc, metrics_selected.KNN.Acc, metrics_selected.RF.Acc]';

Prec = [metrics_all.SVM.Prec, metrics_all.KNN.Prec, metrics_all.RF.Prec, ...
        metrics_selected.SVM.Prec, metrics_selected.KNN.Prec, metrics_selected.RF.Prec]';

Rec  = [metrics_all.SVM.Rec, metrics_all.KNN.Rec, metrics_all.RF.Rec, ...
        metrics_selected.SVM.Rec, metrics_selected.KNN.Rec, metrics_selected.RF.Rec]';

F1s  = [metrics_all.SVM.F1, metrics_all.KNN.F1, metrics_all.RF.F1, ...
        metrics_selected.SVM.F1, metrics_selected.KNN.F1, metrics_selected.RF.F1]';

ClassCol = [Classifiers; Classifiers];

ResultsTable = table(FeatureType, ClassCol, Acc, Prec, Rec, F1s);
disp(ResultsTable);

%% -------------------- Save Results to Excel --------------------
writetable(ResultsTable, 'Results.xlsx');

%% -------------------- Collect Scores for ROC --------------------
scores_all = getCVScores_for_ROC(X, y);
scores_sel = getCVScores_for_ROC(X_selected, y);

%% -------------------- Mean ROC (SVM + KNN + RF) --------------------
fprintf("\nComputing MEAN ROC (SVM + KNN + RF)...\n");

% Collect ROC curves for ALL features
[Xall1,Yall1,~,AUC_all_SVM] = perfcurve(scores_all.SVM.ytrue, scores_all.SVM.score, 1);
[Xall2,Yall2,~,AUC_all_KNN] = perfcurve(scores_all.KNN.ytrue, scores_all.KNN.score, 1);
[Xall3,Yall3,~,AUC_all_RF]  = perfcurve(scores_all.RF.ytrue,  scores_all.RF.score, 1);

% Collect ROC curves for SELECTED features
[Xsel1,Ysel1,~,AUC_sel_SVM] = perfcurve(scores_sel.SVM.ytrue, scores_sel.SVM.score, 1);
[Xsel2,Ysel2,~,AUC_sel_KNN] = perfcurve(scores_sel.KNN.ytrue, scores_sel.KNN.score, 1);
[Xsel3,Ysel3,~,AUC_sel_RF]  = perfcurve(scores_sel.RF.ytrue,  scores_sel.RF.score, 1);

% Interpolation grid
Xgrid = linspace(0,1,400);

% Use safe interpolation to remove duplicates
Y_all_mean = ( safeInterp(Xall1,Yall1,Xgrid) + ...
               safeInterp(Xall2,Yall2,Xgrid) + ...
               safeInterp(Xall3,Yall3,Xgrid) ) ./ 3;

Y_sel_mean = ( safeInterp(Xsel1,Ysel1,Xgrid) + ...
               safeInterp(Xsel2,Ysel2,Xgrid) + ...
               safeInterp(Xsel3,Ysel3,Xgrid) ) ./ 3;

AUC_all_mean = mean([AUC_all_SVM, AUC_all_KNN, AUC_all_RF]);
AUC_sel_mean = mean([AUC_sel_SVM, AUC_sel_KNN, AUC_sel_RF]);

%% -------------------- Plot Mean ROC --------------------
figure('Units','centimeters','Position',[3 3 12 10]);  
hold on; box on;

plot(Xgrid, Y_all_mean, 'k-', 'LineWidth',2.5, ...
    'DisplayName', sprintf('All Features (AUC=%.3f)',AUC_all_mean));

plot(Xgrid, Y_sel_mean, 'k--', 'LineWidth',2.5, ...
    'DisplayName', sprintf('Selected Features (AUC=%.3f)',AUC_sel_mean));

plot([0 1],[0 1],'k:','LineWidth',1.5,'HandleVisibility','off');

xlabel('False Positive Rate','FontSize',12,'FontWeight','bold');
ylabel('True Positive Rate','FontSize',12,'FontWeight','bold');
legend('Location','SouthEast','FontSize',10,'Box','off');
title('Mean ROC (SVM + KNN + RF)','FontSize',13,'FontWeight','bold');

% PDF vector size fixed for Nature
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 10]);
print('ROC_mean_Nature','-dpdf','-r300','-bestfit');
saveas(gcf,'ROC_mean_Nature.png');

fprintf("Mean ROC saved as ROC_mean_Nature.pdf & PNG.\n");

%% =========================================================
% FUNCTIONS
%% =========================================================

function metrics = runClassifiers(X, y)
    k = 5;
    cv = cvpartition(y, 'KFold', k);

    SVM=[]; KNN=[]; RF=[];

    for i = 1:k
        Xtr = X(training(cv,i),:);  
        ytr = y(training(cv,i));
        Xte = X(test(cv,i),:);     
        yte = y(test(cv,i));

        % --- SVM ---
        m = fitcsvm(Xtr,ytr,'KernelFunction','rbf','Standardize',true);
        m = fitPosterior(m);
        yp = predict(m,Xte);
        SVM = [SVM; calc(yte, yp)];

        % --- KNN ---
        m = fitcknn(Xtr,ytr,'NumNeighbors',5,'Standardize',true);
        yp = predict(m,Xte);
        KNN = [KNN; calc(yte, yp)];

        % --- Random Forest ---
        m = fitcensemble(Xtr,ytr,'Method','Bag');
        yp = predict(m,Xte);
        RF = [RF; calc(yte, yp)];
    end

    metrics.SVM = meanStruct(SVM);
    metrics.KNN = meanStruct(KNN);
    metrics.RF  = meanStruct(RF);
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

%% --- safe interpolation for ROC
function Ynew = safeInterp(X, Y, Xgrid)
    [Xunique, idx] = unique(X,'stable');
    Yunique = Y(idx);

    if numel(Xunique)<3
        % fallback linear
        Ynew = interp1([0 1],[0 1], Xgrid,'linear','extrap');
        return;
    end
    Ynew = interp1(Xunique,Yunique,Xgrid,'linear','extrap');
end

%% --- collect scores for ROC
function Scores = getCVScores_for_ROC(X, y)
    cv = cvpartition(y,'KFold',5);
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
        m = fitcsvm(Xtr,ytr,'KernelFunction','rbf','Standardize',true);
        m = fitPosterior(m);
        [~,s] = predict(m,Xte);
        Scores.SVM.ytrue = [Scores.SVM.ytrue; yte];
        Scores.SVM.score = [Scores.SVM.score; s(:,2)];

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
