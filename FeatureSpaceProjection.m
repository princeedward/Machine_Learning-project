function [NewSamples, NewLabels] = FeatureSpaceProjection(Xt_counts, Yt, KeyFeaturesIndex)
NewFeatureSpace = full(Xt_counts(:,KeyFeaturesIndex));
NewSamples = NewFeatureSpace(sum(NewFeatureSpace,2)~=0,:);
NewLabels = Yt(sum(NewFeatureSpace,2)~=0,:);
