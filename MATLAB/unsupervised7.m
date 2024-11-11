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

% 1. Split the dataset into training (70%), validation (15%), and test (15%) sets
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

% 2. Manual PCA Implementation (No Toolbox)
% Step 1: Compute the covariance matrix on the training data
covMatrix = cov(trainData);

% Step 2: Compute eigenvectors and eigenvalues using eig function
[eigenvectors, eigenvalues] = eig(covMatrix);

% Step 3: Sort eigenvalues and eigenvectors in descending order
[sortedEigenvalues, sortIdx] = sort(diag(eigenvalues), 'descend');
sortedEigenvectors = eigenvectors(:, sortIdx);

% Step 4: Select top 'numComponents' eigenvectors (Principal Components)
numComponents = 3; % Choose the number of principal components
trainReducedFeatures = trainData * sortedEigenvectors(:, 1:numComponents);

% Apply PCA to validation and test sets
valReducedFeatures = valData * sortedEigenvectors(:, 1:numComponents);
testReducedFeatures = testData * sortedEigenvectors(:, 1:numComponents);

% 3. K-Means Clustering (No Toolbox) on the training set
numClusters = 3;  % Number of clusters
maxIter = 100;    % Maximum iterations
tol = 1e-4;       % Convergence tolerance

% Initialize centroids randomly
centroids = trainReducedFeatures(randperm(size(trainReducedFeatures, 1), numClusters), :);

% K-Means Iteration on the training data
for iter = 1:maxIter
    % Assign samples to the nearest centroid using Euclidean distance
    distances = zeros(size(trainReducedFeatures, 1), numClusters);
    for i = 1:size(trainReducedFeatures, 1)
        for k = 1:numClusters
            distances(i, k) = sum((trainReducedFeatures(i, :) - centroids(k, :)).^2); % Euclidean distance
        end
    end
    [~, clusterIndices] = min(distances, [], 2);
    
    % Compute new centroids
    newCentroids = zeros(numClusters, size(trainReducedFeatures, 2));
    for k = 1:numClusters
        newCentroids(k, :) = mean(trainReducedFeatures(clusterIndices == k, :), 1);
    end
    
    % Check for convergence (if centroids do not change significantly)
    if norm(newCentroids - centroids) < tol
        disp('Convergence reached.');
        break;
    end
    centroids = newCentroids;
end

% 4. Evaluate clustering on validation and test sets
% Assign clusters to validation data
distancesVal = zeros(size(valReducedFeatures, 1), numClusters);
for i = 1:size(valReducedFeatures, 1)
    for k = 1:numClusters
        distancesVal(i, k) = sum((valReducedFeatures(i, :) - centroids(k, :)).^2); % Euclidean distance
    end
end
[valClusterIndices, ~] = min(distancesVal, [], 2);

% Assign clusters to test data
distancesTest = zeros(size(testReducedFeatures, 1), numClusters);
for i = 1:size(testReducedFeatures, 1)
    for k = 1:numClusters
        distancesTest(i, k) = sum((testReducedFeatures(i, :) - centroids(k, :)).^2); % Euclidean distance
    end
end
[testClusterIndices, ~] = min(distancesTest, [], 2);

% 5. Evaluate the Silhouette Score (Quality of Clustering)
trainSilhouette = silhouette(trainReducedFeatures, clusterIndices);
valSilhouette = silhouette(valReducedFeatures, valClusterIndices);
testSilhouette = silhouette(testReducedFeatures, testClusterIndices);

disp(['Training Silhouette Score: ', num2str(mean(trainSilhouette))]);
disp(['Validation Silhouette Score: ', num2str(mean(valSilhouette))]);
disp(['Test Silhouette Score: ', num2str(mean(testSilhouette))]);

% 6. Plot clustering results for training, validation, and test sets
figure;
subplot(1,4,1);
scatter(trainReducedFeatures(:, 1), trainReducedFeatures(:, 2), 30, clusterIndices, 'filled');
title('Training Set Clustering');
xlabel('Principal Component 1');
ylabel('Principal Component 2');

subplot(1,4,2);
scatter(valReducedFeatures(:, 1), valReducedFeatures(:, 2), 30, valClusterIndices, 'filled');
title('Validation Set Clustering');
xlabel('Principal Component 1');
ylabel('Principal Component 2');

subplot(1,4,3);
scatter(testReducedFeatures(:, 1), testReducedFeatures(:, 2), 30, testClusterIndices, 'filled');
title('Test Set Clustering');
xlabel('Principal Component 1');
ylabel('Principal Component 2');

% 7. Plot accuracy (Silhouette Score) with a centered constant line
subplot(1,4,4);
accuracy = [mean(trainSilhouette), mean(valSilhouette), mean(testSilhouette)];
clusterQuality = accuracy; % Silhouette score as a measure of accuracy
centerLine = ones(size(accuracy)) * 0.5; % Assuming 0.5 as a centered line for ideal score

plot(1:3, accuracy, '-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:3, centerLine, '--r', 'LineWidth', 2); % Center line at 0.5
title('Clustering Accuracy (Silhouette Score)');
xlabel('Dataset');
ylabel('Silhouette Score');
xticks(1:3);
xticklabels({'Train', 'Validation', 'Test'});
legend('Silhouette Score', 'Centered Line (0.5)', 'Location', 'Best');
hold off;


%{
Silhouette Score: Evaluates the quality of clustering for training, validation, and test sets.

Centered Constant Line: A line at 0.5 is added as a baseline or 
expected value for the silhouette score, and the clustering results are compared against this line.

Accuracy Plot: The Silhouette Scores are used as a measure of clustering quality 
(acting as an accuracy metric), plotted for training, validation, and test sets.

%}