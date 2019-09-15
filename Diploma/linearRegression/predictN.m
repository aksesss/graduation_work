function [ y ] = predictN( X, theta, n )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
y = [];
for i = 1:n
    qq = linPredict(X(end,:),theta);
    q = [X(end, 2:end), qq];
    X = [X;q];
    y = [y; qq];
end

end