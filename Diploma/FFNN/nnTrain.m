function [ initial_nn_params_min, nn_cf_cv_min, nn_params_min ] = nnTrain(input_layer_size,  hidden_layer_size, num_labels, X_train, y_train, X_cv, y_cv, costFunction, lambda, iters, options)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for i = 1:iters    
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    
    % Create "short hand" for the cost function to be minimized
%     costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);
    
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
    
    nn_cf_cv = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X_cv, y_cv, lambda) ;            
                 
    if ~exist('nn_params_min')
        initial_nn_params_min = initial_nn_params;
        nn_cf_cv_min = nn_cf_cv;
        nn_params_min = nn_params;
    elseif nn_cf_cv < nn_cf_cv_min
        initial_nn_params_min = initial_nn_params;
        nn_cf_cv_min = nn_cf_cv; 
        nn_params_min = nn_params;
    end
end



end