clc;
clearvars;

% Loading varible data
% Define the folder path containing .mat files
folderPath = 'mlproject/u1';

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(folderPath, '*.mat'));

% Check if there are any .mat files in the folder
if isempty(matFiles)
    error('No .mat files found in the specified folder.');
end

% Loop through each .mat file and load its variables
for k = 1:length(matFiles)
    % Get the full path of the .mat file
    fileName = matFiles(k).name;
    fullFilePath = fullfile(folderPath, fileName);
    
    % Display the name of the file being loaded
    fprintf('Loading %s...\n', fileName);
    
    % Load variables from the .mat file
    data = load(fullFilePath);
    
    % Access variables in the loaded data (each variable can be accessed by name)
    variableNames = fieldnames(data);
    for i = 1:length(variableNames)
        variableName = variableNames{i};
        assignin('base', variableName, data.(variableName));  % Save each variable to the base workspace
    end
end

disp('All .mat files have been loaded successfully.');

% Example data (replace with your actual data)
x1 = Acc_FD_Feat_Vec; % Frequency domain features for user 1
y1 = Acc_TD_Feat_Vec; % Time domain features for user 1

% Check if data is loaded correctly
if isempty(x1) || isempty(y1)
    error('Data for x1 or y1 is missing. Ensure Acc_FD_Feat_Vec and Acc_TD_Feat_Vec are loaded correctly.');
end

% Combine the frequency-domain and time-domain features for each user
combinedFeatures = [x1, y1]; % Combine x1 and y1 horizontally

% Display the size of the combined data
disp(['Combined Features Size: ', num2str(size(combinedFeatures))]);

% Normalize the combined features
combinedFeatures = (combinedFeatures - mean(combinedFeatures, 1)) ./ std(combinedFeatures, [], 1);

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
hiddenLayerSize = 5;  % Size of the hidden layer
autoenc = patternnet(hiddenLayerSize, 'trainlm');  % Use 'patternnet' for a standard neural network

% Configure the Autoencoder
autoenc.trainParam.epochs = 1000;      % Number of epochs
autoenc.trainParam.max_fail = 6;       % Number of validation failures before stopping
autoenc.trainParam.min_grad = 1e-5;    % Gradient tolerance
autoenc.trainParam.showWindow = true;  % Show training window

% Train the Autoencoder
[autoenc, tr] = train(autoenc, trainReducedFeatures', trainReducedFeatures');

% Plot the training performance
figure;
plotperform(tr);

% Get the best validation performance
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

% Display the number of epochs trained
disp(['Number of epochs trained: ', num2str(tr.num_epochs)]);

% Visualize the reconstructed data for the test set
figure;
subplot(1, 2, 1);
plot(testReducedFeatures(1, :));
title('Original Test Data');

subplot(1, 2, 2);
plot(testOutput(1, :));
title('Reconstructed Test Data');
