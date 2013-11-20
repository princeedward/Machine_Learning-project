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
% NumberOfWords = 1000;
% % This is the number of top feature words we want to find from each rate
% level, which tries to get rid of those meaningless words like 'the', 'a'...
% TopWordsNumber = 50;
% ImportantWords = zeros(5,NumberOfWords);
% for i=1:5
%     AccumulateResult = full(sum(Xt_counts(Yt==i,:),1)');
%     [SortedResult,IX] = sort(AccumulateResult,'descend');
%     topWorlds = vocab(IX(1:NumberOfWords));
%     ImportantWords(i,:) = IX(1:NumberOfWords)';
% %     figure
% %     bar(SortedResult(1:NumberOfWords));
% %     set(gca,'xticklabel',topWorlds);
% %     title(['Result for',int2str(i),'stars']);
% end
% EGD = zeros(5,NumberOfWords);
% RealTopWords = cell(5,TopWordsNumber);
% RealTopWordsIndex = zeros(5,TopWordsNumber);
% for i=1:5
%     for n = 1:NumberOfWords
%         [row,col] = find(ImportantWords==ImportantWords(i,n));
%         EGD(i,n) = sum(abs(col-n))+NumberOfWords*(5-length(col));
%     end
%     [SortedEGD,topIX] = sort(EGD(i,:),'descend');
%     SampledWords = ImportantWords(i,:);
%     topIX = topIX(SortedEGD<(3*NumberOfWords));
%     RealTopWords(i,:) = vocab(SampledWords(topIX(1:TopWordsNumber)));
%     RealTopWordsIndex(i,:) = SampledWords(topIX(1:TopWordsNumber));
% end
% KeyFeatures = unique(reshape(RealTopWords,[],1));
% KeyFeaturesIndex = unique(reshape(RealTopWordsIndex,[],1))';
% NewFeatureSpace = full(Xt_counts(:,KeyFeaturesIndex));
% NewSamples = NewFeatureSpace(sum(NewFeatureSpace,2)~=0,:);
% NewLabels = Yt(sum(NewFeatureSpace,2)~=0,:);
%% Runing LR on training data set to get the classifier
% LGclassifier = mnrfit(NewSamples,NewLabels);
% TestNewFeatureSpace = full(Xq_counts(:,KeyFeaturesIndex));
% TestNewSamples = TestNewFeatureSpace(sum(TestNewFeatureSpace,2)~=0,:);
% PredictPos = mnrval(LGclassifier,TestNewSamples);
% maxPos = repmat(max(PredictPos,[],2),1,5);
% PredictLabels = zeros(size(TestNewSamples,1),1);
% for i= 1:size(TestNewSamples,1)
%     PredictLabels(i) = sum((PredictPos(i,:)==max(PredictPos(i,:))).*(1:5));
% end
rates = zeros(5000,1);
rates(sum(TestNewFeatureSpace,2)~=0) = PredictLabels;
RestSamples = find(rates==0);
LabelsForTheOtherGroup = zeros(length(RestSamples),1);
AverageReference=zeros(5,size(Xt_counts,2));
for i=1:5
    AverageReference(i,:) = mean(Xt_counts(Yt==i),1);
end
for i=1:length(RestSamples)
    similarScore = zeros(5,1);
    for n=1:5
        similarScore(n) = sum(bsxfun(@min,Xq_counts(i,:),AverageReference(n,:))');
    end
    LabelsForTheOtherGroup(i) = round(mean(Yt(find(similarScore==max(similarScore)))));
end
profile off
profile viewer
%% Run algorithm
% rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features,...
%                        Xq_additional_features, Yt);

%% Save results to a text file for submission
% dlmwrite('submit.txt',rates,'precision','%d');