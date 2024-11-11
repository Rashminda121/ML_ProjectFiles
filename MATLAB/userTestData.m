clc;
clearvars;

% Example data (replace with your actual data)
x1 = randn(36, 43); % Frequency domain features for user 1
x2 = randn(36, 43); % Frequency domain features for user 2
x3 = randn(36, 43); % Frequency domain features for user 3
x4 = randn(36, 43); % Frequency domain features for user 4
x5 = randn(36, 43); % Frequency domain features for user 5
x6 = randn(36, 43); % Frequency domain features for user 6
x7 = randn(36, 43); % Frequency domain features for user 7
x8 = randn(36, 43); % Frequency domain features for user 8
x9 = randn(36, 43); % Frequency domain features for user 9
x10 = randn(36, 43); % Frequency domain features for user 10

y1 = randn(36, 88); % Time domain features for user 1
y2 = randn(36, 88); % Time domain features for user 2
y3 = randn(36, 88); % Time domain features for user 3
y4 = randn(36, 88); % Time domain features for user 4
y5 = randn(36, 88); % Time domain features for user 5
y6 = randn(36, 88); % Time domain features for user 6
y7 = randn(36, 88); % Time domain features for user 7
y8 = randn(36, 88); % Time domain features for user 8
y9 = randn(36, 88); % Time domain features for user 9
y10 = randn(36, 88); % Time domain features for user 10

% Combine the frequency-domain and time-domain features for each user
combinedFeatures = [
    [x1, y1];
    [x2, y2];
    [x3, y3];
    [x4, y4];
    [x5, y5];
    [x6, y6];
    [x7, y7];
    [x8, y8];
    [x9, y9];
    [x10, y10]
];

% Labels for each user (1 to 10) - 36 samples for each user
labels = repelem(1:10, 36)';  % Create label vector

% Normalize the combined features
combinedFeatures = (combinedFeatures - mean(combinedFeatures)) ./ std(combinedFeatures);

% Split the data: 70% training, 15% validation, 15% testing
numSamples = length(labels);
indices = randperm(numSamples);
trainIdx = indices(1:round(0.7 * numSamples));
valIdx = indices(round(0.7 * numSamples) + 1:round(0.85 * numSamples));
testIdx = indices(round(0.85 * numSamples) + 1:end);

trainData = combinedFeatures(trainIdx, :);
trainLabels = labels(trainIdx);

valData = combinedFeatures(valIdx, :);
valLabels = labels(valIdx);

testData = combinedFeatures(testIdx, :);
testLabels = labels(testIdx);

% Neural Network Setup
hiddenLayerSize = 50; % Increase the number of hidden neurons
net = feedforwardnet(hiddenLayerSize);

% Set the training function (use 'trainbr' for Bayesian regularization)
net.trainFcn = 'trainbr'; % This can help prevent overfitting and improve convergence

% Define the output layer activation function for classification (softmax for probability distribution)
net.layers{end}.transferFcn = 'softmax';  % Use softmax for classification

% Set the maximum number of epochs
net.trainParam.epochs = 100; % Limit the number of epochs (e.g., 100 epochs)

% Set up training, validation, and test data in the network
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainIdx;
net.divideParam.valInd = valIdx;
net.divideParam.testInd = testIdx;

% Train the Neural Network using the training data
[net, tr] = train(net, combinedFeatures', labels');

% Test the Neural Network using the test data
predictedLabelsRaw = net(testData')';

% Convert network outputs to class labels (find the max value across outputs)
[~, predictedLabels] = max(predictedLabelsRaw, [], 2);

% Calculate accuracy of the predictions
accuracy = sum(predictedLabels == testLabels) / length(testLabels) * 100;

% Display the accuracy
disp(['Accuracy of User Identity Verification: ', num2str(accuracy), '%']);

% Display confusion matrix
testLabelsCat = categorical(testLabels, 1:10, {'User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7', 'User 8', 'User 9', 'User 10'});
predictedLabelsCat = categorical(predictedLabels, 1:10, {'User 1', 'User 2', 'User 3', 'User 4', 'User 5', 'User 6', 'User 7', 'User 8', 'User 9', 'User 10'});
figure;
confusionchart(testLabelsCat, predictedLabelsCat);
title('Confusion Matrix for Test Data');
