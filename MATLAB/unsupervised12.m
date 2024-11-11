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

% Combine the frequency-domain and time-domain features
combinedFeatures = [Acc_FD_Feat_Vec, Acc_TD_Feat_Vec_subset];
disp(['Combined Features Size: ', num2str(size(combinedFeatures))]);

% Normalize the combined features
combinedFeatures = (combinedFeatures - mean(combinedFeatures, 1)) ./ std(combinedFeatures, [], 1);

% Split the dataset into training (70%), validation (15%), and test (15%) sets
numSamples = size(combinedFeatures, 1);
numTrain = round(0.7 * numSamples);
numVal = round(0.15 * numSamples);
numTest = numSamples - numTrain - numVal;

% Shuffle and split the data
shuffledIndices = randperm(numSamples);
trainData = combinedFeatures(shuffledIndices(1:numTrain), :);
valData = combinedFeatures(shuffledIndices(numTrain+1:numTrain+numVal), :);
testData = combinedFeatures(shuffledIndices(numTrain+numVal+1:end), :);

% Display sizes of splits
disp(['Train Data Size: ', num2str(size(trainData))]);
disp(['Validation Data Size: ', num2str(size(valData))]);
disp(['Test Data Size: ', num2str(size(testData))]);

% Apply PCA (Optional)
covMatrix = cov(trainData);
[eigenvectors, eigenvalues] = eig(covMatrix);
[~, sortIdx] = sort(diag(eigenvalues), 'descend');
sortedEigenvectors = eigenvectors(:, sortIdx);
numComponents = 3;
trainReducedFeatures = trainData * sortedEigenvectors(:, 1:numComponents);
valReducedFeatures = valData * sortedEigenvectors(:, 1:numComponents);
testReducedFeatures = testData * sortedEigenvectors(:, 1:numComponents);

% Display PCA reduced feature sizes
disp(['Train Reduced Features Size: ', num2str(size(trainReducedFeatures))]);
disp(['Validation Reduced Features Size: ', num2str(size(valReducedFeatures))]);
disp(['Test Reduced Features Size: ', num2str(size(testReducedFeatures))]);

% Define Autoencoder Architecture with MSE
hiddenLayerSize = 5;
autoenc = patternnet(hiddenLayerSize, 'trainlm');
autoenc.performFcn = 'mse';

% Configure Autoencoder
autoenc.trainParam.epochs = 1000;
autoenc.trainParam.max_fail = 6;
autoenc.trainParam.min_grad = 1e-5;
autoenc.trainParam.showWindow = true;

% Train the Autoencoder
[autoenc, tr] = train(autoenc, trainReducedFeatures', trainReducedFeatures');

% Plot training performance
figure;
plotperform(tr);
title('Training Performance');
drawnow;

% Get best validation performance
bestValidationPerformance = min(tr.vperf);
disp(['Best Validation Performance: ', num2str(bestValidationPerformance)]);

% Validate the model
valOutput = autoenc(valReducedFeatures');
valLoss = immse(valOutput', valReducedFeatures);
disp(['Validation Loss: ', num2str(valLoss)]);

% Test the model
testOutput = autoenc(testReducedFeatures');
testLoss = immse(testOutput', testReducedFeatures);
disp(['Test Loss: ', num2str(testLoss)]);

% Display epochs trained
disp(['Number of epochs trained: ', num2str(tr.num_epochs)]);

% Visualize reconstructed data for the test set
figure;
subplot(1, 2, 1);
plot(testReducedFeatures(1, :));
title('Original Test Data');
xlabel('Feature Index');
ylabel('Feature Value');
drawnow;

subplot(1, 2, 2);
plot(testOutput(1, :));
title('Reconstructed Test Data');
xlabel('Feature Index');
ylabel('Reconstructed Value');
drawnow;
