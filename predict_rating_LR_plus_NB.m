function rates = predict_rating_LR_plus_NB(classifier, KeyFeaturesIndex,Xt_counts, Xq_counts, Xt_additional_features,...
                                Xq_additional_features, Yt)
% Returns the predicted ratings, given wordcounts and additional features.
%
% Usage:
%
%   RATES = PREDICT_RATING(XT_COUNTS, XQ_COUNTS, XT_ADDITIONAL_FEATURES, ...
%                         XQ_ADDITIONAL_FEATURES, YT);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of wordcount and additional features and produces a
% ranking matrix as explained in the project overview.
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 10 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE
N = size(Xq_counts, 1);

% rates = int8(ones(N,1));
rates = zeros(N,1);

%% This part generate the possiblity of each word appear in each rate
AverageReference=zeros(5,size(Xt_counts,2));
RatePos = zeros(5,1);
for i=1:5
    AverageReference(i,:) = mean(Xt_counts(Yt==i,:),1);
    RatePos(i) = mean(Yt==i);
end
%% This part calculates the predict labels
TestNewFeatureSpace = full(Xq_counts(:,KeyFeaturesIndex));
TestNewSamples = TestNewFeatureSpace(sum(TestNewFeatureSpace,2)~=0,:);
PredictPos = mnrval(classifier,TestNewSamples);
% maxPos = repmat(max(PredictPos,[],2),1,5);
PredictLabels = zeros(size(TestNewSamples,1),1);
for i= 1:size(TestNewSamples,1)
    PredictLabels(i) = sum((PredictPos(i,:)==max(PredictPos(i,:))).*(1:5));
end
rates(sum(TestNewFeatureSpace,2)~=0) = PredictLabels;
RestSamples = find(rates==0);
LabelsForTheOtherGroup = zeros(length(RestSamples),1);
for i=1:length(RestSamples)
    WordAppearance = find(Xq_counts(RestSamples(i),:));
    similarScore = zeros(5,1);
    for n=1:5
        WordPossibility = AverageReference(n,:);
        similarScore(n) = RatePos(n)*prod(WordPossibility(WordAppearance));
    end
    LabelsForTheOtherGroup(i) = round(mean(find(similarScore==max(similarScore))));
end
rates(RestSamples) = LabelsForTheOtherGroup;
end
