function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

s1 = sigmoid(X*theta);
J = -y'*log(s1) - (1-y)'*(log(1 - s1));
J = J./m
J = J + theta(2:length(theta))'*theta(2:length(theta)) * lambda / [2 * m] 

grad1 = (s1 - y)'*X./m;
grad2 = (s1 - y)'*X./m + theta' .* lambda ./m;

grad = [grad1(1), grad2(2:length(grad2))]';





% =============================================================

end
