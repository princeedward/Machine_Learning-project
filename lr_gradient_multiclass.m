function [grad] = lr_gradient_multiclass(X, Y, w, C)
% Compute the Logistic Regression gradient.
%
% Usage:
%
%    [GRAD] = LR_GRADIENT(X, Y, W, C)
%
% X is a N x P matrix of N examples with P features each. Y is a N x 1 vector
% of (-1, +1) class labels. W is a 1 x P weight vector. C is the regularization
% parameter. Computes the gradient w.r.t. W of the regularized logistic
% regression objective and returns a 1 x P vector GRAD.
%
% SEE ALSO
%   LR_TRAIN, LR_TEST

% YOUR CODE GOES HERE
ep = exp(-Y.*(X*w'));
ep(isinf(ep)) = realmax;
P = 1./(1+ ep);
grad =sum(repmat(Y,1,size(X,2)).*X.*repmat((1-P),1,size(X,2)),1) -C*w;