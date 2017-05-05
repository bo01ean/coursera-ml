function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly
J = 0;


L = zeros(n + 1);
L(2:end, 2:end) = eye(n);

m = length(y); % number of training examples
####
#tmp = -y' * log(h) - ( 1 - y )' * log(1 - h);

#tmp
#tmpTheta = theta(3:end, 1);
#regularizationTerm =  sum( lambda * (tmpTheta .^ 2));

#J = 1 / ( 2 * m) * (tmp + regularizationTerm);
#J


#[J, grad] = costFunction(theta, X, y);

% [J, grad] = costFunction(theta, X, y);
%(X' * X + lambda * L)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

alpha = 1;

A = alpha / m;

B = -y' * log(h) - ( 1 - y )' * log(1 - h);

C = sum(theta(2:end) .^ 2);

J = (A * B) + ((lambda / (2 * m)) * C);


grad = zeros(size(theta));
grad = (alpha / m) * X' * (sigmoid(X * theta) - y);

  for j = 2:length(grad)
    grad(j) = grad(j) + ((lambda / m) * (theta(j)));
  endfor
end
