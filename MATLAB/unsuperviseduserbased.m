clc;
clearvars;

% Example data (replace with your actual data)
x1 = randn(36, 43); % Frequency domain features for user 1

y1 = randn(36, 88); % Time domain features for user 1

% Combine the frequency-domain and time-domain features for each user
combinedFeatures = [x1, y1]; % Combine x1 and y1 horizontally

% Check the size of the combined dataset
inputDimensions = size(combinedFeatures); % the input dimensions refer to the number of features, the size of each sample, and the number of samples in your dataset.
disp(['Input Dimensions: ', inputDimensions]);

% Get the number of columns
[inputDimensionsRows, inputDimensionsColumns] = size(inputDimensions);

% Display the number of columns
disp(['Number of Input Dimensions columns: ', inputDimensionsColumns]);

% Normalize the combined features
combinedFeatures = (combinedFeatures - mean(combinedFeatures)) ./ std(combinedFeatures);

% Split the dataset into training (70%), validation (15%), and test (15%) sets
numSamples = size(combinedFeatures, 1);
numTrain = round(0.7 * numSamples);   % 70% for training
numVal = round(0.15 * numSamples);    % 15% for validation
numTest = numSamples - numTrain - numVal;  % Remaining for test

% Randomly shuffle the indices
shuffledIndices = randperm(numSamples);

% Split the data  % : all columns
trainData = combinedFeatures(shuffledIndices(1:numTrain), :);
valData = combinedFeatures(shuffledIndices(numTrain+1:numTrain+numVal), :);
testData = combinedFeatures(shuffledIndices(numTrain+numVal+1:end), :);

% Apply PCA (Optional) % Principal Component Analysis
covMatrix = cov(trainData);
[eigenvectors, eigenvalues] = eig(covMatrix);
[sortedEigenvalues, sortIdx] = sort(diag(eigenvalues), 'descend');
sortedEigenvectors = eigenvectors(:, sortIdx);
numComponents = 3;
trainReducedFeatures = trainData * sortedEigenvectors(:, 1:numComponents);
valReducedFeatures = valData * sortedEigenvectors(:, 1:numComponents);
testReducedFeatures = testData * sortedEigenvectors(:, 1:numComponents);

% Define the Autoencoder Architecture
hiddenLayerSize = 5;  % Size of the hidden layer % number of neurons
autoenc = feedforwardnet(hiddenLayerSize, 'trainlm');  % 'trainlm' uses Levenberg-Marquardt for faster training
dropoutRate = 0.25;     % Fraction of neurons to drop (25%)

% Define the layers for the autoencoder
layers = [
    imageInputLayer([1, 1, inputDimensionsColumns], 'Name', 'input')   % Input layer (adjust size as needed)
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'hidden')        % Hidden layer
    dropoutLayer(dropoutRate, 'Name', 'dropout')                   % Dropout layer
    fullyConnectedLayer(1, 'Name', 'output') % Output layer (reconstruction)
    regressionLayer('Name', 'regression')   % Regression layer for continuous output
    ];

% Create the layer graph
lgraph = layerGraph(layers);


% epocs
numberofEpocs=300;

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs', numberofEpocs, ...
    'MiniBatchSize', 32, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

trainTargets = trainData;  % For autoencoders, targets are the same as inputs

% Train the Autoencoder
autoenc = trainNetwork(trainData, trainTargets, lgraph, options);

% Display the network structure
disp(autoenc.Layers);


% Configure the Autoencoder
autoenc.layers{1}.transferFcn = 'logsig'; % Log-sigmoid activation function
autoenc.layers{2}.transferFcn = 'purelin'; % Linear activation for output layer

% Set training parameters
autoenc.trainParam.epochs = numberofEpocs;  % Number of epochs
autoenc.trainParam.max_fail = 6;  % Number of validation failures before stopping
autoenc.trainParam.min_grad = 1e-5; % Gradient tolerance
autoenc.trainParam.showWindow = true; % Don't show the training window % show training window

% Train the Autoencoder
[autoenc, tr] = train(autoenc, trainReducedFeatures', trainReducedFeatures');

% Plot the training progress %
figure;
plotperform(tr);


% Get the best validation performance
bestValidationPerformance = min(tr.best_vperf);

% Add a horizontal line at the best validation performance
hold on;
yline(bestValidationPerformance, 'r--', ['Best Validation Performance = ' num2str(bestValidationPerformance)]);
hold off;

% Validate the model %
valOutput = autoenc(valReducedFeatures');
valLoss = immse(valOutput, valReducedFeatures');
disp(['Validation Loss: ', num2str(valLoss)]);

% Test the model %
testOutput = autoenc(testReducedFeatures');
testLoss = immse(testOutput, testReducedFeatures');
disp(['Test Loss: ', num2str(testLoss)]);

% display the number of epocs % how many times the model has seen and learned from the entire dataset.
disp(['Number of epochs trained: ', num2str(tr.num_epochs)]);

% Visualize the reconstructed data for test set
figure;
subplot(1,2,1); % This divides the figure window into a 1x2 grid (1 row, 2 columns). The 1 in the third argument means the plot will be placed in the first position (left side) of the grid.
plot(testReducedFeatures(1,:));
title('Original Test Data');

subplot(1,2,2);
plot(testOutput(1,:));
title('Reconstructed Test Data');
