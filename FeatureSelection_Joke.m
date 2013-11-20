function [KeyFeaturesIndex,RealTopWords] = FeatureSelection_Joke(Xt_counts, Yt, vocab, NumberOfWords, TopWordsNumber)
ImportantWords = zeros(5,NumberOfWords);
for i=1:5
    AccumulateResult = full(sum(Xt_counts(Yt==i,:),1)');
    [SortedResult,IX] = sort(AccumulateResult,'descend');
%     topWorlds = vocab(IX(1:NumberOfWords));
    ImportantWords(i,:) = IX(1:NumberOfWords)';
%     figure
%     bar(SortedResult(1:NumberOfWords));
%     set(gca,'xticklabel',topWorlds);
%     title(['Result for',int2str(i),'stars']);
end
EMD = zeros(5,NumberOfWords);
RealTopWords = cell(5,TopWordsNumber);
RealTopWordsIndex = zeros(5,TopWordsNumber);
for i=1:5
    for n = 1:NumberOfWords
        [~,col] = find(ImportantWords==ImportantWords(i,n));
        EMD(i,n) = sum((n-col));
    end
    [SortedEGD,topIX] = sort(EMD(i,:),'descend');
    SampledWords = ImportantWords(i,:);
%     topIX = topIX(SortedEGD<(FilterThreshold*NumberOfWords));
    RealTopWords(i,:) = vocab(SampledWords(topIX(1:TopWordsNumber)));
    RealTopWordsIndex(i,:) = SampledWords(topIX(1:TopWordsNumber));
end
% KeyFeaturesWords = unique(reshape(RealTopWords,[],1));
KeyFeaturesIndex = unique(reshape(RealTopWordsIndex,[],1))';