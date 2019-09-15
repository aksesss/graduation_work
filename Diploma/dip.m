addpath('data','preprocessing','plotting')
addpath('linearRegression')
T = readtable('AABA_2006-01-01_to_2018-01-01.csv');
X1 = (1:size(T,1))';
X1(:,2) = T{1:end,1}.Day;
X1(:,3) = T{1:end,1}.Month;
X1(:,4) = T{1:end,1}.Year;

%select features
X = X1(:,1);
y = T{1:end,5};


%%display first feature
%plotData(X(:,1),y,'day', 'cost')
% Scale features and set them to zero mean
[X, mux, sigmax] = featureNormalize(X);
[y, muy, sigmay] = featureNormalize(y);
plotData(X(:,1),y,'day', 'cost')


alpha = 0.1;
iterations = 500;

% Init Theta and Run Gradient Descent 
theta = zeros(size(X,2)+1, 1);
theta = gradientDescentMulti(X, y, theta, alpha, iterations);


%/////
% Plot the linear fit
hold on; % keep previous plot visible
hyp = [ones(size(X,1), 1) X]*theta;
plot(X, hyp, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure
cost = computeCostMulti(X,y,theta)


%///
%
%Lets do some featuremapping to the 6th degree
%X = X1(1:100);
X = X1(1:100,:);
%X = mapFeature(4,X(:,1), X(:,3));
%X = [X, X1];
y=y(1:size(X,1));
[X, mux, sigmax] = featureNormalize(X);
alpha = 0.1;
iterations = 1000;

% Init Theta and Run Gradient Descent 
theta = zeros(size(X,2)+1, 1);
theta = gradientDescentMulti(X, y, theta, alpha, iterations);%//


%//
% Plot the linear fit
plotData(X(:,1),y,'day', 'cost')
plotData(X(:,1),y,'day', 'cost');
hold on; % keep previous plot visible

hyp = [ones(size(X,1), 1) X]*theta;
plot(X(:,1), hyp, '-','Color','b')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

%//cost = computeCostMulti(X,y,theta)
