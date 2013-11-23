function K = kernel_intersection(X, X2)
% Evaluates the Histogram Intersection Kernel
%
% Usage:
%
%    K = KERNEL_INTERSECTION(X, X2)
%
% For a N x D matrix X and a M x D matrix X2, computes a M x N kernel
% matrix K where K(i,j) = k(X(i,:), X2(j,:)) and k is the histogram
% intersection kernel.

n = size(X,1);
m = size(X2,1);

% HINT: Transpose the sparse data matrix X, so that you can operate over columns. Sparse
% column operations in matlab are MUCH faster than row operations.

% YOUR CODE GOES HERE.
% A = repmat(reshape(X2',[],1),n,1);
% B = reshape(repmat(X',m,1),[],1);
% Raw = sum(reshape(min(A-B,[],2),size(x,2),[]),1);
% K = reshape(Raw,m,n);
A = X2';
B = X';
for i = 1:m
    for j = 1:n
        K(i,j) = sum(min([A(:,i),B(:,j)],[],2));
    end
end

K = full(K);