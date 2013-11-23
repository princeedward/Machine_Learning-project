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
% % % This is the top frequently appeared words in each rate level
% NumberOfWords = 1000;
% % % This is the number of top feature words we want to find from each rate
% % level, which tries to get rid of those meaningless words like 'the', 'a'...
% TopWordsNumber = 50;
% % FilterThreshold = 4;
%% New feature slection
% ImportantWords = zeros(5,NumberOfWords);
% for i=1:5
%     AccumulateResult = full(sum(Xt_counts(Yt==i,:),1)');
%     [SortedResult,IX] = sort(AccumulateResult,'descend');
% %     topWorlds = vocab(IX(1:NumberOfWords));
%     ImportantWords(i,:) = IX(1:NumberOfWords)';
% %     figure
% %     bar(SortedResult(1:NumberOfWords));
% %     set(gca,'xticklabel',topWorlds);
% %     title(['Result for',int2str(i),'stars']);
% end
% EMD = zeros(5,NumberOfWords);
% RealTopWords = cell(5,TopWordsNumber);
% RealTopWordsIndex = zeros(5,TopWordsNumber);
% for i=1:5
%     for n = 1:NumberOfWords
%         [~,col] = find(ImportantWords==ImportantWords(i,n));
%         EMD(i,n) = sum(abs(n-col));
%     end
%     [SortedEGD,topIX] = sort(EMD(i,:),'descend');
%     SampledWords = ImportantWords(i,:);
% %     topIX = topIX(SortedEGD<(FilterThreshold*NumberOfWords));
%     RealTopWords(i,:) = vocab(SampledWords(topIX(1:TopWordsNumber)));
%     RealTopWordsIndex(i,:) = SampledWords(topIX(1:TopWordsNumber));
% end
% % KeyFeaturesWords = unique(reshape(RealTopWords,[],1));
% KeyFeaturesIndex = unique(reshape(RealTopWordsIndex,[],1))';
%% Find the new feature space
% load FrequencyData.mat
% [KeyFeaturesIndex1,RealTopWords1] = FeatureSelection_EMD(Xt_counts_frequency, Yt, vocab, NumberOfWords, TopWordsNumber, 1);
% [KeyFeaturesIndex2,RealTopWords2] = FeatureSelection_EMD(Xt_counts_frequency, Yt, vocab, NumberOfWords, TopWordsNumber, 2);
% [KeyFeaturesIndex3,RealTopWords3] = FeatureSelection_EMD(Xt_counts_frequency, Yt, vocab, NumberOfWords, TopWordsNumber, 3);
% [KeyFeaturesIndex4,RealTopWords4] = FeatureSelection_EMD(Xt_counts_frequency, Yt, vocab, NumberOfWords, TopWordsNumber, 4);
% [KeyFeaturesIndex5,RealTopWords5] = FeatureSelection_EMD(Xt_counts_frequency, Yt, vocab, NumberOfWords, TopWordsNumber, 5);
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
% -------------------- See what happened to the words ---------------------
% [KeyFeaturesIndex6,RealTopWords6] = FeatureSelection_Joke(Xt_counts_frequency, Yt, vocab, NumberOfWords, TopWordsNumber);
% [NewSamples] = WholeFeatureReducedData(Xt_counts, [8155, 34482,22361,55850,29938,54696,21965,2885,14203,35064]);
% for i = 1:5
%     AccumulateResult = full(sum(NewSamples(Yt==i,:),1)');
%     figure
%     bar(AccumulateResult);
% end
% -------------------- The statistic frequency ----------------------------
% wordCounts = full(sum(Xt_counts',1));
% tic
% Xt_counts_frequency = Xt_counts;
% for i = 1:size(Xt_counts,1)
%     Xt_counts_frequency(i,:) = (Xt_counts_frequency(i,:)'/wordCounts(i))';
% end
% toc
% display('Feature selection complete');
%% Normalize all the stuffs classification
% WholeFeatureSpace = [KeyFeaturesIndex1, KeyFeaturesIndex2, KeyFeaturesIndex3, KeyFeaturesIndex4, KeyFeaturesIndex5, KeyFeaturesIndex6];
% FeaturesStar = cell(5,1);
% for i = 1:5
%     FeaturesStar{i} = unique(WholeFeatureSpace(i,:));
% end
% StarWords = zeros(size(Xt_counts,1),5);
% for i = 1:5
%     StarWords(:,i) = sum(WholeFeatureReducedData(Xt_counts,FeaturesStar{i}),2);
% end
% % StarWords(:,6) = ones(size(StarWords,1),1);
% WordsCountInNewSpace = sum(StarWords,2);
% StarWords = StarWords./repmat(WordsCountInNewSpace,1,5);
% StarWords(isnan(StarWords)) = 0;
% --------------- Naive Bayes ---------------------------------------------
% O1 = NaiveBayes.fit(StarWords,Yt);
% TrainLabels = O1.predict(StarWords);
% --------------- Linear Regression ---------------------------------------
% mdl = fitlm(StarWords,Yt);
% TrainLabelsLR = predict(mdl,StarWords);
% --------------- Support Vector ------------------------------------------
% - The reason it does not converge is the dimension of the feature space
% - is less than the the number of points
% k = @(x,x2) kernel_intersection(x, x2);
% Yt5 = Yt;
% Yt5(Yt~=5) = 0;
% results5 = kernel_libsvm(Xt_counts, Yt5, Xt_counts, Yt5, k);% ERROR RATE OF INTERSECTION KERNEL GOES HERE
% Yt4R = Yt;
% Yt4R(Yt4R<4) = 0;
% Yt4 = Yt(Yt5==0);
% Yt4(Yt4~=4) = 0;
% results4 = kernel_libsvm(Xt_counts(Yt5==0), Yt4, Xt_counts(Yt5==0), Yt4, k);% ERROR RATE OF INTERSECTION KERNEL GOES HERE
% Yt3R = Yt;
% Yt3R(Yt3R<3) = 0;
% Yt3 = Yt(YT4R==0);
% Yt3(Yt3~=3) = 0;
% results3 = kernel_libsvm(Xt_counts(YT4R==0), Yt3, Xt_counts(YT4R==0), Yt3, k);% ERROR RATE OF INTERSECTION KERNEL GOES HERE
% Yt2 = Yt(Yt3R==0);
% Yt2(Yt2~=2) = 1;
% results2 = kernel_libsvm(Xt_counts(Yt3R==0), Yt2, Xt_counts(Yt3R==0), Yt2, k);% ERROR RATE OF INTERSECTION KERNEL GOES HERE
% TrainLabelsSVM = zeros(size(Yt,1),1);
% TrainLabelsSVM(results5.yhat==5) = 5;
% TrainLabelsSVM(TrainLabelsSVM==0) = results4.yhat;
% TrainLabelsSVM(TrainLabelsSVM==0) = results3.yhat;
% TrainLabelsSVM(TrainLabelsSVM==0) = results2.yhat;
% 
% results = kernel_libsvm(Xt_counts, Yt, Xt_counts, Yt, k);% ERROR RATE OF INTERSECTION KERNEL GOES HERE
%% Boost method
% [NewSamples] = WholeFeatureReducedData(Xt_counts, hugeFeatureSpace);
% totalStump1 = fitensemble(NewSamples,Yt,'TotalBoost',150,'Tree','kfold',5);
% [NewSamplesTest] = WholeFeatureReducedData(Xq_counts, hugeFeatureSpace);
% Label1 = predict(totalStump1,Xq_counts);
% display('Boost on reduced data set complete');
% figure
% plot(kfoldLoss(totalStump1,'mode','cumulative'),'r.');
% totalStump2 = fitensemble(Xt_counts,Yt,'TotalBoost',200,'Tree','kfold',5);
% Label2 = predict(totalStump2,Xq_counts);
% display('Boost on whole data set complete');
% figure
% plot(kfoldLoss(totalStump2,'mode','cumulative'),'b.');
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

% PredictLabels1 = zeros(size(PredictPos1,1),1);
% for i= 1:size(PredictPos1,1)
%     PredictLabels1(i) = sum((PredictPos1(i,:)==max(PredictPos1(i,:))).*(1:5));
% end
% PredictLabels2 = zeros(size(PredictPos1,1),1);
% for i= 1:size(PredictPos1,1)
%     PredictLabels2(i) = sum((PredictPos2(i,:)==max(PredictPos2(i,:))).*(1:5));
% end
% PredictLabels3 = zeros(size(PredictPos1,1),1);
% for i= 1:size(PredictPos1,1)
%     PredictLabels3(i) = sum((PredictPos3(i,:)==max(PredictPos3(i,:))).*(1:5));
% end
% PredictLabels4 = zeros(size(PredictPos1,1),1);
% for i= 1:size(PredictPos1,1)
%     PredictLabels4(i) = sum((PredictPos4(i,:)==max(PredictPos4(i,:))).*(1:5));
% end
% PredictLabels5 = zeros(size(PredictPos1,1),1);
% for i= 1:size(PredictPos1,1)
%     PredictLabels5(i) = sum((PredictPos5(i,:)==max(PredictPos5(i,:))).*(1:5));
% end
% -------------------- Linear boost ---------------------------------------
% Predictors = [PredictLabels1 PredictLabels2 PredictLabels3 PredictLabels4 PredictLabels5];
% b_normal = glmfit(Predictors,Yt,'normal');
% yhat_normal = glmval(b_normal,Predictors,'identity');
% yhat_normal = round(yhat_normal);
% yhat_normal(yhat_normal<1) = 1;
% yhat_normal(yhat_normal>5) = 5;
% error_rate_normal = mean(yhat_normal~=Yt);
% RMSE_normal = sqrt(mean((yhat_normal-Yt).^2));
% 
% b_poisson = glmfit(Predictors,Yt,'poisson');
% yhat_poisson = glmval(b_poisson,Predictors,'log');
% yhat_poisson = round(yhat_poisson);
% yhat_poisson(yhat_poisson<1) = 1;
% yhat_poisson(yhat_poisson>5) = 5;
% error_rate_poisson = mean(yhat_poisson~=Yt);
% RMSE_poisson = sqrt(mean((yhat_poisson-Yt).^2));
% 
% b_binomial = glmfit(Predictors,Yt,'inverse gaussian');
% yhat_binomial = glmval(b_binomial,Predictors,-2);
% yhat_binomial = round(yhat_binomial);
% yhat_binomial(yhat_binomial<1) = 1;
% yhat_binomial(yhat_binomial>5) = 5;
% error_rate_binomial = mean(yhat_binomial~=Yt);
% RMSE_binomial = sqrt(mean((yhat_binomial-Yt).^2));

% nb = NaiveBayes.fit(Predictors, Yt);
% yhat_nb = predict(nb,Predictors);
% error_rate_nb = mean(yhat_nb~=Yt);
% RMSE_nb = sqrt(mean((yhat_nb-Yt).^2));
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