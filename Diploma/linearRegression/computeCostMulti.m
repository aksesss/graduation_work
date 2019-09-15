function J = computeCostMulti(X, y, theta, lambda)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
for i = (1:m)
   J = J + (X(i,:)*theta - y(i))^2; 
end
J = J/2/m;
% add = lambda*theta/m;
% add(1) = 0;
% J = J + add;
% =========================================================================

end
