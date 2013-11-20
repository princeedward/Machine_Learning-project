% clear;
% load ../data/review_dataset.mat
% 
% Xt_counts = train.counts;
% Yt = train.labels;
% Xq_counts = quiz.counts;
% 
% initialize_additional_features;
%% This part of code is used for testing
profile on
%% Feature selection
% % This is the top frequently appeared words in each rate level
NumberOfWords = 1000;
% % This is the number of top feature words we want to find from each rate
% level, which tries to get rid of those meaningless words like 'the', 'a'...
TopWordsNumber = 50;
FilterThreshold = 4;
%% New feature slection
[KeyFeaturesIndex,RealTopWords] = FeatureSelection_Joke(Xt_counts, Yt, vocab, NumberOfWords, TopWordsNumber);
%% Find the new feature space
% [KeyFeaturesIndex1,RealTopWords1] = FeatureSelection_EGD(Xt_counts, Yt, vocab, NumberOfWords, TopWordsNumber, 1);
% [KeyFeaturesIndex2,RealTopWords2] = FeatureSelection_EGD(Xt_counts, Yt, vocab, NumberOfWords, TopWordsNumber, 2);
% [KeyFeaturesIndex3,RealTopWords3] = FeatureSelection_EGD(Xt_counts, Yt, vocab, NumberOfWords, TopWordsNumber, 3);
% [KeyFeaturesIndex4,RealTopWords4] = FeatureSelection_EGD(Xt_counts, Yt, vocab, NumberOfWords, TopWordsNumber, 4);
% [KeyFeaturesIndex5,RealTopWords5] = FeatureSelection_EGD(Xt_counts, Yt, vocab, NumberOfWords, TopWordsNumber, 5);
% hugeFeatureSpace = unique([KeyFeaturesIndex1,KeyFeaturesIndex2,KeyFeaturesIndex3,KeyFeaturesIndex4,KeyFeaturesIndex5]);
% [NewSamples1, NewLabels1] = FeatureSpaceProjection(Xt_counts, Yt, KeyFeaturesIndex1);
% [NewSamples2, NewLabels2] = FeatureSpaceProjection(Xt_counts, Yt, KeyFeaturesIndex2);
% [NewSamples3, NewLabels3] = FeatureSpaceProjection(Xt_counts, Yt, KeyFeaturesIndex3);
% [NewSamples4, NewLabels4] = FeatureSpaceProjection(Xt_counts, Yt, KeyFeaturesIndex4);
% [NewSamples5, NewLabels5] = FeatureSpaceProjection(Xt_counts, Yt, KeyFeaturesIndex5);
% [NewSamples1] = WholeFeatureReducedData(Xt_counts, KeyFeaturesIndex1);
% [NewSamples2] = WholeFeatureReducedData(Xt_counts, KeyFeaturesIndex2);
% [NewSamples3] = WholeFeatureReducedData(Xt_counts, KeyFeaturesIndex3);
% [NewSamples4] = WholeFeatureReducedData(Xt_counts, KeyFeaturesIndex4);
% [NewSamples5] = WholeFeatureReducedData(Xt_counts, KeyFeaturesIndex5);
%% First Layer of Neural Network: Logistic regression
% LGclassifier1 = mnrfit(NewSamples1, Yt);
% LGclassifier2 = mnrfit(NewSamples2, Yt);
% LGclassifier3 = mnrfit(NewSamples3, Yt);
% LGclassifier4 = mnrfit(NewSamples4, Yt);
% LGclassifier5 = mnrfit(NewSamples5, Yt);
%% Second Layer of Neural Network: Logistic regression
% PredictPos1 = mnrval(LGclassifier1,NewSamples1);
% PredictPos2 = mnrval(LGclassifier2,NewSamples2);
% PredictPos3 = mnrval(LGclassifier3,NewSamples3);
% PredictPos4 = mnrval(LGclassifier4,NewSamples4);
% PredictPos5 = mnrval(LGclassifier5,NewSamples5);
% LGclassifierSec = mnrfit([PredictPos1,PredictPos2,PredictPos3,PredictPos4,PredictPos5], Yt);
% PredictPos = mnrval(LGclassifierSec,[PredictPos1,PredictPos2,PredictPos3,PredictPos4,PredictPos5]);
% PredictLabels = zeros(size(PredictPos,1),1);
% for i= 1:size(PredictPos,1)
%     PredictLabels(i) = sum((PredictPos(i,:)==max(PredictPos(i,:))).*(1:5));
% end
%% Save the classifier
% save('Net_LR_Classifier.mat','KeyFeaturesIndex1','KeyFeaturesIndex2','KeyFeaturesIndex3','KeyFeaturesIndex4','KeyFeaturesIndex5','LGclassifier1','LGclassifier2','LGclassifier3','LGclassifier4','LGclassifier5','LGclassifierSec');
%% This part calculates the training error
% Trainrates = predict_rating_LR_plus_NB(LGclassifier, KeyFeaturesIndex,Xt_counts, Xt_counts, Xt_additional_features,...
%                                 Xq_additional_features, Yt);
% Trainrates = PredictLabels;
% TrainError = mean(Trainrates~=Yt);
% TrainRMSE = sqrt(mean((Trainrates-Yt).^2));
%% This part calculates the predict labels
% TestNewFeatureSpace = full(Xq_counts(:,KeyFeaturesIndex));
% TestNewSamples = TestNewFeatureSpace(sum(TestNewFeatureSpace,2)~=0,:);
% PredictPos = mnrval(LGclassifier,TestNewSamples);
% maxPos = repmat(max(PredictPos,[],2),1,5);
% PredictLabels = zeros(size(TestNewSamples,1),1);
% for i= 1:size(TestNewSamples,1)
%     PredictLabels(i) = sum((PredictPos(i,:)==max(PredictPos(i,:))).*(1:5));
% end
% rates = zeros(5000,1);
% rates(sum(TestNewFeatureSpace,2)~=0) = PredictLabels;
% RestSamples = find(rates==0);
% LabelsForTheOtherGroup = zeros(length(RestSamples),1);
% for i=1:length(RestSamples)
%     WordAppearance = find(Xq_counts(RestSamples(i),:));
%     similarScore = zeros(5,1);
%     for n=1:5
%         WordPossibility = AverageReference(n,:);
%         similarScore(n) = RatePos(n)*prod(WordPossibility(WordAppearance));
%     end
%     LabelsForTheOtherGroup(i) = round(mean(find(similarScore==max(similarScore))));
% end
% rates(RestSamples) = LabelsForTheOtherGroup;
% [NewSamples1] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex1);
% [NewSamples2] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex2);
% [NewSamples3] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex3);
% [NewSamples4] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex4);
% [NewSamples5] = WholeFeatureReducedData(Xq_counts, KeyFeaturesIndex5);
% PredictPos1 = mnrval(LGclassifier1,NewSamples1);
% PredictPos2 = mnrval(LGclassifier2,NewSamples2);
% PredictPos3 = mnrval(LGclassifier3,NewSamples3);
% PredictPos4 = mnrval(LGclassifier4,NewSamples4);
% PredictPos5 = mnrval(LGclassifier5,NewSamples5);
% PredictPos = mnrval(LGclassifierSec,[PredictPos1,PredictPos2,PredictPos3,PredictPos4,PredictPos5]);
% PredictLabels = zeros(size(PredictPos,1),1);
% for i= 1:size(PredictPos,1)
%     PredictLabels(i) = sum((PredictPos(i,:)==max(PredictPos(i,:))).*(1:5));
% end
% rates = PredictLabels;
profile off
profile viewer
%% Run algorithm
% rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
%                        Xq_additional_features, Yt);

%% Save results to a text file for submission
% dlmwrite('submit.txt',rates,'precision','%d');