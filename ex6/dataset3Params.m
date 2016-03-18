function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Set of testing values from the exersice description
C_val = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_val = [0.01 0.03 0.1 0.3 1 3 10 30];
loop_limit = length(C_val);
predictions_error = zeros(loop_limit);

for i=1:loop_limit
    for j=1:loop_limit
        C = C_val(i);
        sigma = sigma_val(j);
        % Obtain the best theta values for a given C and sigma using
        % the training algorithm
        learned_theta = svmTrain(X, y, C, ...
            @(x1, x2) gaussianKernel(x1, x2, sigma));
        % Get the predictions using the learned thetas and
        % the cross validation set
        predictions = svmPredict(learned_theta, Xval);
        % Calculate the prediction error comparing the predictions with
        % the valitation expected values
        predictions_error(i, j) = mean(double (predictions ~= yval));
    end
end

% printf('predictions errors:\n');
% predictions_error;

% Identify the minimum error, we use nested min(min(...)) in order to find
% the row with the minimum values (inner min) and next the colum with the
% minumum value (outer min)
min_error = min(min(predictions_error));
% Find the position of the minimum prediction error to identify which
% combination of C and sigma generated it.
[i j] = find(predictions_error == min_error);
C = C_val(i);
sigma = sigma_val(j);

% =========================================================================

end
