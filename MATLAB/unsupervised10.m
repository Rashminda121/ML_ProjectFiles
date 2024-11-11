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

% Normalize the combined features
combinedFeatures = (combinedFeatures - mean(combinedFeatures)) ./ std(combinedFeatures);

% Split the dataset into training (70%), validation (15%), and test (15%) sets
numSamples = size(combinedFeatures, 1);
numTrain = round(0.7 * numSamples);   % 70% for training
numVal = round(0.15 * numSamples);    % 15% for validation
numTest = numSamples - numTrain - numVal;  % Remaining for test

% Randomly shuffle the indices
shuffledIndices = randperm(numSamples);

% Split the data
trainData = combinedFeatures(shuffledIndices(1:numTrain), :);
valData = combinedFeatures(shuffledIndices(numTrain+1:numTrain+numVal), :);
testData = combinedFeatures(shuffledIndices(numTrain+numVal+1:end), :);

% Apply PCA (Optional)
covMatrix = cov(trainData);
[eigenvectors, eigenvalues] = eig(covMatrix);
[sortedEigenvalues, sortIdx] = sort(diag(eigenvalues), 'descend');
sortedEigenvectors = eigenvectors(:, sortIdx);
numComponents = 3;
trainReducedFeatures = trainData * sortedEigenvectors(:, 1:numComponents);
valReducedFeatures = valData * sortedEigenvectors(:, 1:numComponents);
testReducedFeatures = testData * sortedEigenvectors(:, 1:numComponents);

% Define the Autoencoder Architecture
hiddenLayerSize = 50;  % Size of the hidden layer
autoenc = feedforwardnet(hiddenLayerSize, 'trainlm');  % 'trainlm' uses Levenberg-Marquardt for faster training

% Configure the Autoencoder
autoenc.layers{1}.transferFcn = 'logsig'; % Log-sigmoid activation function
autoenc.layers{2}.transferFcn = 'purelin'; % Linear activation for output layer

% Set training parameters
autoenc.trainParam.epochs = 100;  % Number of epochs
autoenc.trainParam.max_fail = 6;  % Number of validation failures before stopping
autoenc.trainParam.min_grad = 1e-5; % Gradient tolerance
autoenc.trainParam.showWindow = false; % Don't show the training window

% Train the Autoencoder
[autoenc, tr] = train(autoenc, trainReducedFeatures', trainReducedFeatures');

% Plot the training progress
figure;
plotperform(tr);

% Validate the model
valOutput = autoenc(valReducedFeatures');
valLoss = immse(valOutput, valReducedFeatures');
disp(['Validation Loss: ', num2str(valLoss)]);

% Test the model
testOutput = autoenc(testReducedFeatures');
testLoss = immse(testOutput, testReducedFeatures');
disp(['Test Loss: ', num2str(testLoss)]);

% Visualize the reconstructed data for test set
figure;
subplot(1,2,1);
plot(testReducedFeatures(1,:));
title('Original Test Data');

subplot(1,2,2);
plot(testOutput(1,:));
title('Reconstructed Test Data');
