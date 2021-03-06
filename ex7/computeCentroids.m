function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X)

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% We iterate on each centroid to find the means of each one's data points
for i=1:K
    % We identify which points belong to each centroid by assigning 1 (true) to
    % the row where the current centroid has a data point. Remember that idx
    % has the same number of rows as X.
    centroid_points = idx == i;
    % We calculate the mean of every centroid. By multiplying by centroid_points
    % each value of X we make zero each point that doesn't belong to
    % the current centroid.
    centroids(i,:) = sum(X .* centroid_points) / sum(centroid_points);
end

% =============================================================


end

