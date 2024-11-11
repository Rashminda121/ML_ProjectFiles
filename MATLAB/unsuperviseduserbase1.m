clc;
clearvars;


% Define the folder path containing .mat files
folderPath = 'mlproject/u1';
matFiles = dir(fullfile(folderPath, '*.mat'));

% Check if there are any .mat files in the folder
if isempty(matFiles)
    error('No .mat files found in the specified folder.');
end

% Load .mat files and assign variables to the base workspace
for k = 1:length(matFiles)
    fileName = matFiles(k).name;
    fullFilePath = fullfile(folderPath, fileName);
    fprintf('Loading %s...\n', fileName);
    data = load(fullFilePath);
    variableNames = fieldnames(data);
    for i = 1:length(variableNames)
        variableName = variableNames{i};
        assignin('base', variableName, data.(variableName));
    end
end

disp('All .mat files have been loaded successfully.');
disp(' ');  % Blank line

% Check for required variables and display error if missing
if ~exist('Acc_FD_Feat_Vec', 'var') || ~exist('Acc_TD_Feat_Vec', 'var')
    error('Data for Acc_FD_Feat_Vec or Acc_TD_Feat_Vec is missing.');
end

% Getting mean value
% Calculate the mean of each column in Acc_TD_Feat_Vec
columnMeans = mean(Acc_TD_Feat_Vec);

% Find columns that are closest to or higher than the mean value
% Sort columns in descending order based on their mean values
[~, sortedIndices] = sort(columnMeans, 'descend');

% Select the top 43 columns based on the sorted mean values
selectedIndices = sortedIndices(1:43);

% Extract these columns from Acc_TD_Feat_Vec
Acc_TD_Feat_Vec_subset = Acc_TD_Feat_Vec(:, selectedIndices);


% Combine the frequency-domain and time-domain features for each user
combinedFeatures = [Acc_FD_Feat_Vec, Acc_TD_Feat_Vec_subset];
disp(['Combined Features Size: ', num2str(size(combinedFeatures))]);
disp(' ');  % Blank line

% Check the size of the combined dataset
inputDimensions = size(combinedFeatures); % The input dimensions refer to the number of features, the size of each sample, and the number of samples in your dataset.
disp(['Input Dimensions: ', inputDimensions]);

% Get the number of rows and columns
[inputDimensionsRows, inputDimensionsColumns] = size(inputDimensions);

% Display the number of rows and columns
disp(['Number of Input Dimensions rows: ', num2str(inputDimensionsRows), ', columns: ', num2str(inputDimensionsColumns)]);
disp(' ');  % Blank line

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

% Define the Autoencoder Architecture with multiple hidden layers
hiddenLayerSizes = [5, 3, 5];  % Three layers with 5, 3, and 5 neurons respectively
autoenc = feedforwardnet(hiddenLayerSizes, 'trainlm');  % 'trainlm' uses Levenberg-Marquardt for faster training

% Specify the performance function to Mean Squared Error (MSE)
autoenc.performFcn = 'mse';

% Set the number of epochs for training
numberofEpochs = 1000;

% Display the network structure % Display if needed
%disp(autoenc);
%disp(' ');  % Blank line for spacing

%{

An autoencoder is designed to encode input data into a compressed (latent) 
representation and then decode it back to a reconstruction of the original input. It has two main components:

%}

% Configure the transfer functions for the layers
for i = 1:length(hiddenLayerSizes)
    autoenc.layers{i}.transferFcn = 'logsig';  % Log-sigmoid for hidden layers
end
autoenc.layers{end}.transferFcn = 'purelin';  % Linear activation for output layer

% Set training parameters
autoenc.trainParam.epochs = numberofEpochs;      % Number of training epochs
autoenc.trainParam.max_fail = 6;                 % Max validation failures before stopping
autoenc.trainParam.min_grad = 1e-5;              % Minimum gradient tolerance
autoenc.trainParam.showWindow = true;            % Show the training window during training

% Train the Autoencoder
[autoenc, tr] = train(autoenc, trainReducedFeatures', trainReducedFeatures');

% Plot the training progress %
figure;
plotperform(tr);
title('Training Performance');
drawnow;


% Get the best validation performance
bestValidationPerformance = min(tr.best_vperf);
disp(['Best Validation Performance: ', num2str(bestValidationPerformance)]);
disp(' ');  % Blank line

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
disp(' ');  % Blank line

% display the number of epocs % how many times the model has seen and learned from the entire dataset.
disp(['Number of epochs trained: ', num2str(tr.num_epochs)]);
disp(' ');  % Blank line for spacing

% Visualize the reconstructed data for test set
figure;
subplot(1,2,1); % This divides the figure window into a 1x2 grid (1 row, 2 columns). The 1 in the third argument means the plot will be placed in the first position (left side) of the grid.
plot(testReducedFeatures(1,:));
title('Original Test Data');
xlabel('Feature Index');
ylabel('Feature Value');
drawnow;

subplot(1,2,2);
plot(testOutput(1,:));
title('Reconstructed Test Data');
xlabel('Feature Index');
ylabel('Reconstructed Value');
drawnow;

