function [NewSamples] = WholeFeatureReducedData_sparse(Xt_counts, KeyFeaturesIndex)
NewSamples = Xt_counts(:,KeyFeaturesIndex);