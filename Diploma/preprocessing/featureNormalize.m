function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.



n = size(X,2);
mu =  mean(X(:,1));
sigma =  std(X(:,1));

for i = 2:n
    mu =  [mu, mean(X(:,i))];
    sigma =  [sigma, std(X(:,i))];
end

X_norm = (X(:,1) - mu(1))/sigma(1);

for i = 2:n
    X_norm = [X_norm,  (X(:,i) - mu(i))/sigma(i)];
end

% ============================================================

end
