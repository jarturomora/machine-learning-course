function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features)

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% Calculate cost function where R(i,j) = 1. By multiplying R by
% the cost function formula (X * Theta' - Y).^2 I only keep the movies where
% a user has made a recommendation.
J = sum(sum(R .* (X * Theta' - Y).^2)) / 2;

% Calculate gradient descent for X on every movie where there is a
% rating from a user

for i=1:num_movies
    % find the index of users that have rated a particular movie
    user_index = find(R(i, :) == 1);
    % select the thetas for those users that have rated a particular movie
    user_thetas = Theta(user_index, :);
    % select the ratings of the users that have rated a particular movie
    movie_Ys = Y(i, user_index);
    % Calculate the gradient for the current movie
    X_grad(i, :) = (X(i, :) * user_thetas' - movie_Ys) * user_thetas;
end


% Calculate gradient descent for theta of every user that has rated a movie.
for i=1:num_users
    % find the index of movies that have a rating.
    movie_index = find(R(:, i) == 1);
    % select the features of movies that have a rating
    movie_X = X(movie_index, :);
    % selectr the movies that have rated the current user
    movie_Ys = Y(movie_index, i);
    % calculate the gradient descent for the current user
    Theta_grad(i, :) = (movie_X * Theta(i, :)' - movie_Ys)' * movie_X;
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
