% Plot the linear fit
plotData(1:size(y,1),y.*mx_y,'day', 'cost');
hold on; % keep previous plot visible
hyp = [ones(size(X_train,1), 1) X_train]*theta;
hyp = hyp.*mx_y;
plot(m_train(1):m_train(2), hyp, '-')

hyp_cv = [ones(size(X_cv,1), 1) X_cv]*theta;
hyp_cv = hyp_cv.*mx_y;
plot(m_cv(1):m_cv(2),hyp_cv, '-');

hyp_test = [ones(size(X_test,1), 1) X_test]*theta;
hyp_test = hyp_test.*mx_y;
plot(m_test(1):m_test(2),hyp_test, '-');

legend('Training data', 'Linear regression' , 'CV set', 'Test set')
hold off % don't overlay any more plots on this figure