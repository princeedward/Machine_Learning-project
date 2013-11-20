function [NewSamples] = WholeFeatureReducedData(Xt_counts, KeyFeaturesIndex)
NewSamples = full(Xt_counts(:,KeyFeaturesIndex));