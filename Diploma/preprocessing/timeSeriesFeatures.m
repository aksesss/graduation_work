function [X,y1] = timeSeriesFeatures( y, n, k )
%UNTITLED Summary of this function goes here
%   Reshape Time series to
% n - cout of days before present

if exist('k')
    if k == 0
        y = log(y(2:end) ./ y(1:end-1));
    elseif k == 1
        y =(y(2:end) - y(1:end-1));   
    end
end

m = size(y,1);

X = zeros(m-n,n);

for i = 1:m-n
    X(i,:) = y(i:n+i-1)'; 
end   

y1 = y(n+1:end);
end