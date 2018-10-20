function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
tests = [.01, .03, .1, .3, 1, 3, 10, 30]
data = zeros(64, 3)

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
data_row = 0
for Ctest = tests
  for sigmatest = tests
    data_row = data_row + 1
    model = svmTrain(X, y, Ctest, @(x1, x2) gaussianKernel(x1, x2, sigmatest))
    predictions = svmPredict(model, Xval)
    error = mean(double(predictions ~= yval))
    data(data_row, :) = [Ctest, sigmatest, error]
  endfor
endfor

[min_val, min_row] = min(data(:,3))
C = data(min_row, 1)
sigma = data(min_row, 2)




% =========================================================================

end
