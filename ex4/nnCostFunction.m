function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% --------------------------------------
% Part 1: Feedforward the neural network
% --------------------------------------
% Calculate de activation functions for the neural network
X = [ones(m, 1), X];
A1 = X;
z2 = A1 * Theta1';
A2 = sigmoid(z2);
A2 = [ones(m, 1), A2];
z3 = A2 * Theta2';
A3 = sigmoid(z3); % h_theta

% Calculate the cost function, we make 'k' additions for the 'm' samples in Jk
for k = 1: num_labels
    % Check if we find a match with y
    yk = y == k;
    h_theta = A3(:, k);
    Jk = sum(-yk .* log(h_theta) - (1 - yk) .* log(1 - h_theta)) / m;
    J = J + Jk;
end

% Regularization of cost function
J = J + (lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2)) ...
    + sum(sum(Theta2(:, 2:end) .^ 2))));

% -----------------------------------------------
% Part 2: Implement the backpropagation algorithm
%         for a three layer network.
% -----------------------------------------------
for t = 1:m
    % Calculate delta_3 for each of the classes
    for k = 1:num_labels
        yk = y(t) == k;
        delta_3(k) = A3(t, k) - yk;
    end

    % Calculate delta_2 for the hidden layer
    delta_2 = Theta2' * delta_3' .* sigmoidGradient([1, z2(t, :)])';
    delta_2 = delta_2(2:end); % Remove bias element

    % Acummulate the gradient for this example
    Theta1_grad = Theta1_grad + delta_2 * A1(t, :);
    Theta2_grad = Theta2_grad + delta_3' * A2(t, :);
end

% Obtaing the unregularized gradient
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularized neural network
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
