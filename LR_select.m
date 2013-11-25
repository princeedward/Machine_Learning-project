%% Fast SMLR on different features
%% Initialization
% clear;
% load ../data/review_dataset.mat
% load features_selected_from_fisher_algo.mat

% Xt_counts = train.counts;
% Yt = train.labels;
% Xq_counts = quiz.counts;

% initialize_additional_features;
%% Words Stemmer
% [NewSamples] = WholeFeatureReducedData_sparse(Xt_counts, fspace.fList(1:1000));
% SVM = liblinear_train(Yt, Xt_counts);
% [predicted_label_class1, accuracy_class1, decision_values_class1] = liblinear_predict(Yt, Xt_counts, SVM);
% [Rates] = liblinear_predict(ones(5000,1), Xq_counts, SVM);
%% Train on SMLR by selecting features
% ------------------ Converting Y -----------------------------------------
% Y1 = Yt==1;
% Y2 = Yt==2;
% Y3 = Yt==3;
% Y4 = Yt==4;
% Y5 = Yt==5;
% Ytr = double([Y1,Y2,Y3,Y4,Y5]);
% % ------------------ Select features to form a new sample -----------------
% [NewSamples] = WholeFeatureReducedData_sparse(Xt_counts, fspace.fList(1:1000));
% [w, args, log_posterior, wasted, saved] = smlr(NewSamples, Ytr);

%% Doing Cross Validation
% numberOfFeatures = 900:50:1500;
% count = 1;
% for i = 1:length(numberOfFeatures)
%     RMSEacc(i) = 0;
%     for m = 1:20
%         [Train, Test] = crossvalind('HoldOut', length(Yt), 0.2);
%         TrainSet = Xt_counts(Train,:);
%         TrainLables = Yt(Train);
%         TestSet = Xt_counts(Test,:);
%         TestLabels = Yt(Test);
%         TrainSetResample = WholeFeatureReducedData_sparse(TrainSet, fspace.fList(1:numberOfFeatures(i)));
%         TestSetResample = WholeFeatureReducedData_sparse(TestSet, fspace.fList(1:numberOfFeatures(i)));
%         SVM(i,m) = liblinear_train(TrainLables, TrainSetResample);
%         [predicted_label, accuracy(i,m), ~] = liblinear_predict(TestLabels, TestSetResample, SVM(i,m));
%         RMSE(i,m) = sqrt(mean((TestLabels-predicted_label).^2));
%         RMSEacc(i) = RMSEacc(i)+RMSE(i,m);
%         count = count + 1;
%     end
%     RMSEacc(i) = RMSEacc(i)/20;
% end
% save('SVMcross.mat');
%% Do the boost on Training data by applying SVM
% The ideal number of features for SVM is 1100 from cross validation
% The feature space is fisher features
numberOfFeatures = 1100;
for i = 1:100
    D_now = ones(20000,1)/20000;
    [Train, Test] = crossvalind('HoldOut', length(Yt), 0.2);
    TrainSet = Xt_counts(Train,:);
    TrainLables = Yt(Train);
    TestSet = Xt_counts(Test,:);
    TestLabels = Yt(Test);
    TrainSetResample = WholeFeatureReducedData_sparse(TrainSet, fspace.fList(1:numberOfFeatures));
    TestSetResample = WholeFeatureReducedData_sparse(TestSet, fspace.fList(1:numberOfFeatures));
    display('Begin the loop!!');
    for T = 1:50
        ActualWeight = D_now*20000;
        for m = 1:20000
            TrainSetResample(m,:) = (ActualWeight(m)*TrainSetResample(m,:)')';
        end
        SVM(T) = liblinear_train(TrainLables, TrainSetResample);
        [predicted_label, ~, ~] = liblinear_predict(TrainLables, TrainSetResample, SVM(T));
        [D_now, alpha(T)] = AdaboostUpdate(D_now, predicted_label, TrainLables,5);
        Predict_init = zeros(5000,1);
        [predicted_labelTest(:,T), ~, ~] = liblinear_predict(TestLabels, TestSetResample, SVM(T));
        Predict_init = [Predict_init, predicted_labelTest(:,T)];
        
        Prediction_Matrix = [sum(repmat(alpha(1:T),5000,1).*(Predict_init==1),2),sum(repmat(alpha(1:T),5000,1).*(Predict_init==2),2),sum(repmat(alpha(1:T),5000,1).*(Predict_init==3),2),sum(repmat(alpha(1:T),5000,1).*(Predict_init==4),2),sum(repmat(alpha(1:T),5000,1).*(Predict_init==5),2)];
        Prediction_Test = sum((Prediction_Matrix==repmat(max(Prediction_Matrix,[],2),1,5)).*repmat(1:5,5000,1),2);
        display(['The RSME after',num2str(T),'times boost is']);
        RSMEs(i,T) = sqrt(mean((Prediction_Test-TestLabels).^2))
    end
end