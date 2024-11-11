clc;
clearvars;

% Loading variable data
folderPath = 'mlproject/u1';
matFiles = dir(fullfile(folderPath, '*.mat'));

% Check if there are any .mat files in the folder
if isempty(matFiles)
    error('No .mat files found in the specified folder.');
end

% Loop through each .mat file and load its variables
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

% Example data
x1 = Acc_FD_Feat_Vec; % Frequency domain features
y1 = Acc_TD_Feat_Vec; % Time domain features

% Check if data is loaded correctly
if isempty(x1) || isempty(y1)
    error('Data for x1 or y1 is missing. Ensure Acc_FD_Feat_Vec and Acc_TD_Feat_Vec are loaded correctly.');
end

% Combine the frequency-domain and time-domain features
combinedFeatures = [x1, y1];
disp(['Combined Features Size: ', num2str(size(combinedFeatures))]);

% Normalize the combined features
combinedFeatures = (combinedFeatures - mean(combinedFeatures, 1)) ./ std(combinedFeatures, [], 1);

% Split the dataset
numSamples = size(combinedFeatures, 1);
numTrain = round(0.7 * numSamples);
numVal = round(0.15 * numSamples);
numTest = numSamples - numTrain - numVal;

% Shuffle and split the data
shuffledIndices = randperm(numSamples);
trainData = combinedFeatures(shuffledIndices(1:numTrain), :);
valData = combinedFeatures(shuffledIndices(numTrain+1:numTrain+numVal), :);
testData = combinedFeatures(shuffledIndices(numTrain+numVal+1:end), :);

% Check split sizes
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

% Check PCA reduced feature sizes
disp(['Train Reduced Features Size: ', num2str(size(trainReducedFeatures))]);
disp(['Validation Reduced Features Size: ', num2str(size(valReducedFeatures))]);
disp(['Test Reduced Features Size: ', num2str(size(testReducedFeatures))]);

% Define Autoencoder Architecture
hiddenLayerSize = 5;
autoenc = patternnet(hiddenLayerSize, 'trainlm');

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
drawnow;

subplot(1, 2, 2);
plot(testOutput(1, :));
title('Reconstructed Test Data');
drawnow;
