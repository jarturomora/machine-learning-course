function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

samples = length(X);

% We iterate on each example to find its best value of C
for i=1:samples
    % Making the assumption that the initial distance between X(i) and
    % mu(i) is infinite, so any shorter distance using all the centroids
    % will be the best.
    distance = inf;
    % Now we iterate on each centroid looking for the shortests distance
    % for each x(i).
    for j=1:K
        centroid_distance = norm(X(i, :) - centroids(j, :));
        % do I find a shorther distance?
        if (centroid_distance < distance)
            distance = centroid_distance;
            % The current shortest distance is the centroid at position j
            idx(i) = j;
        end
    end

end

% =============================================================

end

