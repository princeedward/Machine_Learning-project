function [D_new, alpha] = AdaboostUpdate(D_old, Yhat, Y, K)
% This update method based on the paper 'Multi-class AdaBoost', Zhu, J from
% Stanford
% D_OLD is a n by 1 vector of distributions
% Yhat is a n by 1 vector of predicted labels
% Y is a n by 1 vector of labels
epsilon = sum(D_old.*(Yhat~=Y));
alpha = 1/2*log((1-epsilon)/epsilon) + log(K-1);
updateDirection = double(Yhat==Y);
updateDirection(updateDirection==0)=-1;
D_new = D_old.*exp(-alpha*updateDirection);
D_new = D_new/sum(D_new);