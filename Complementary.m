%% Initialization
clear;
load ../data/review_dataset.mat
% load Net_LR_Classifier.mat
% load PredictLabels_LR.mat

Xt_counts = train.counts;
Yt = train.labels;
Xq_counts = quiz.counts;

initialize_additional_features;
%% Weaker Learner SVM
% --------------- Train on classifier 1 -----------------------------------
Yt_class1 = Yt;
Yt_class1(PredictLabels1~=Yt) = 0;
Yt_class1(PredictLabels1==Yt) = 1;
SVM_class1 = liblinear_train(Yt_class1, Xt_counts);
[predicted_label_class1, accuracy_class1, decision_values_class1] = liblinear_predict(Yt_class1, Xt_counts, SVM_class1);
% --------------- Train on classifier 2 -----------------------------------
Yt_class2 = Yt;
Yt_class2(PredictLabels2~=Yt) = 0;
Yt_class2(PredictLabels2==Yt) = 1;
SVM_class2 = liblinear_train(Yt_class2, Xt_counts);
[predicted_label_class2, accuracy_class2, decision_values_class2] = liblinear_predict(Yt_class2, Xt_counts, SVM_class2);
% --------------- Train on classifier 3 -----------------------------------
Yt_class3 = Yt;
Yt_class3(PredictLabels3~=Yt) = 0;
Yt_class3(PredictLabels3==Yt) = 1;
SVM_class3 = liblinear_train(Yt_class3, Xt_counts);
[predicted_label_class3, accuracy_class3, decision_values_class3] = liblinear_predict(Yt_class3, Xt_counts, SVM_class3);
% --------------- Train on classifier 4 -----------------------------------
Yt_class4 = Yt;
Yt_class4(PredictLabels4~=Yt) = 0;
Yt_class4(PredictLabels4==Yt) = 1;
SVM_class4 = liblinear_train(Yt_class4, Xt_counts);
[predicted_label_class4, accuracy_class4, decision_values_class4] = liblinear_predict(Yt_class4, Xt_counts, SVM_class4);
% --------------- Train on classifier 4 -----------------------------------
Yt_class5 = Yt;
Yt_class5(PredictLabels5~=Yt) = 0;
Yt_class5(PredictLabels5==Yt) = 1;
SVM_class5 = liblinear_train(Yt_class5, Xt_counts);
[predicted_label_class5, accuracy_class5, decision_values_class5] = liblinear_predict(Yt_class5, Xt_counts, SVM_class5);
% --------------- Prediction On Training Data -----------------------------
ClassCat = [predicted_label_class1, predicted_label_class2, predicted_label_class3, predicted_label_class4, predicted_label_class5];
ClassPredict = [PredictLabels1, PredictLabels2, PredictLabels3, PredictLabels4, PredictLabels5];
rates = zeros(length(Yt),1);
for i = 1:length(Yt)
    poolLabels = ClassPredict(i,:);
    rates(i) = round(mean(poolLabels(ClassCat(i,:)==1)));
end
rates(isnan(rates)) = round(mean(ClassPredict(isnan(rates),:),2));
ErrorRate = mean(rates~=Yt);
RMSE = sqrt(mean((rates-Yt).^2));
% --------------- Prediction On Quiz Data ---------------------------------
[NewSamplesQ1] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex1);
[NewSamplesQ2] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex2);
[NewSamplesQ3] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex3);
[NewSamplesQ4] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex4);
[NewSamplesQ5] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex5);
PredictPosQ1 = mnrval(LGclassifier1,NewSamplesQ1);
PredictPosQ2 = mnrval(LGclassifier2,NewSamplesQ2);
PredictPosQ3 = mnrval(LGclassifier3,NewSamplesQ3);
PredictPosQ4 = mnrval(LGclassifier4,NewSamplesQ4);
PredictPosQ5 = mnrval(LGclassifier5,NewSamplesQ5);

PredictLabelsQ1 = zeros(size(PredictPosQ1,1),1);
for i= 1:size(PredictPosQ1,1)
    PredictLabelsQ1(i) = sum((PredictPosQ1(i,:)==max(PredictPosQ1(i,:))).*(1:5));
end
PredictLabelsQ2 = zeros(size(PredictPosQ1,1),1);
for i= 1:size(PredictPosQ1,1)
    PredictLabelsQ2(i) = sum((PredictPosQ2(i,:)==max(PredictPosQ2(i,:))).*(1:5));
end
PredictLabelsQ3 = zeros(size(PredictPosQ1,1),1);
for i= 1:size(PredictPosQ1,1)
    PredictLabelsQ3(i) = sum((PredictPosQ3(i,:)==max(PredictPosQ3(i,:))).*(1:5));
end
PredictLabelsQ4 = zeros(size(PredictPosQ1,1),1);
for i= 1:size(PredictPosQ1,1)
    PredictLabelsQ4(i) = sum((PredictPosQ4(i,:)==max(PredictPosQ4(i,:))).*(1:5));
end
PredictLabelsQ5 = zeros(size(PredictPosQ1,1),1);
for i= 1:size(PredictPosQ1,1)
    PredictLabelsQ5(i) = sum((PredictPosQ5(i,:)==max(PredictPosQ5(i,:))).*(1:5));
end
ClassPredictQ = [PredictLabelsQ1, PredictLabelsQ2, PredictLabelsQ3, PredictLabelsQ4, PredictLabelsQ5];

YtestOcc = ones(size(PredictLabelsQ5));
[predicted_label_classQ1] = liblinear_predict(YtestOcc, Xq_counts, SVM_class1);
[predicted_label_classQ2] = liblinear_predict(YtestOcc, Xq_counts, SVM_class2);
[predicted_label_classQ3] = liblinear_predict(YtestOcc, Xq_counts, SVM_class3);
[predicted_label_classQ4] = liblinear_predict(YtestOcc, Xq_counts, SVM_class4);
[predicted_label_classQ5] = liblinear_predict(YtestOcc, Xq_counts, SVM_class5);
ClassCatQ = [predicted_label_classQ1, predicted_label_classQ2, predicted_label_classQ3, predicted_label_classQ4, predicted_label_classQ5];

Qrates = zeros(length(PredictLabelsQ1),1);
for i = 1:length(PredictLabelsQ1)
    poolLabels = ClassPredictQ(i,:);
    Qrates(i) = round(mean(poolLabels(ClassCatQ(i,:)==1)));
end
Qrates(isnan(Qrates)) = round(mean(ClassPredictQ(isnan(Qrates),:),2));
