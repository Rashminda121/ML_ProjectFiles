% Load or assign user data: 
% x1 to x10 (Frequency domain features) and y1 to y10 (Time domain features)

clc, clearvars ;

% Example data (replace these with actual datasets for your users)
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
labels = repelem(1:5, 36)';

% Normalize the combined features
% Normalize the data to have zero mean and unit variance
combinedFeatures = (combinedFeatures - mean(combinedFeatures)) ./ std(combinedFeatures);

% Neural Network Setup: 
% Define a feedforward neural network with 1 hidden layer and 10 neurons
hiddenLayerSize = 10; % Number of hidden neurons
net = feedforwardnet(hiddenLayerSize);

% Set the training function (Levenberg-Marquardt backpropagation is fast and commonly used for small datasets)
net.trainFcn = 'trainlm';

% Train the Neural Network using the combined features (input) and user labels (output)
[net, tr] = train(net, combinedFeatures', labels');

% View the neural network
view(net);

% Test the Neural Network using the same data (for simplicity, using the same data for both training and testing)
predictedLabels = net(combinedFeatures')';

% Evaluate the performance
% Calculate accuracy of the predictions
accuracy = sum(predictedLabels == labels) / length(labels) * 100;
disp(['Accuracy of User Identity Verification: ', num2str(accuracy), '%']);

% Confusion Matrix to see detailed classification results
figure;
confusionchart(labels, predictedLabels);
title('Confusion Matrix for User Identity Verification');

% Optionally, plot the performance curve to visualize training progress
figure;
plotperform(tr);
