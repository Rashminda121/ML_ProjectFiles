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

% Apply PCA (Manually)
covMatrix = cov(trainData);
[eigenvectors, eigenvalues] = eig(covMatrix);
[sortedEigenvalues, sortIdx] = sort(diag(eigenvalues), 'descend');
sortedEigenvectors = eigenvectors(:, sortIdx);
numComponents = 3;
trainReducedFeatures = trainData * sortedEigenvectors(:, 1:numComponents);
valReducedFeatures = valData * sortedEigenvectors(:, 1:numComponents);
testReducedFeatures = testData * sortedEigenvectors(:, 1:numComponents);

% Initialize parameters
numClusters = 3;  % Number of clusters
maxIter = 100;    % Maximum iterations
tol = 1e-4;       % Convergence tolerance
epochs = 20;      % Train for multiple epochs

% Tracking Silhouette Scores for visualization
trainSilhouetteScores = zeros(epochs, 1);
valSilhouetteScores = zeros(epochs, 1);

% Function to train K-Means and update centroids
function [centroids, clusterIndices] = trainKMeans(trainData, numClusters, maxIter, tol)
    centroids = trainData(randperm(size(trainData, 1), numClusters), :);  % Initialize centroids randomly
    for iter = 1:maxIter
        % Assign samples to the nearest centroid using Euclidean distance
        distances = zeros(size(trainData, 1), numClusters);
        for i = 1:size(trainData, 1)
            for k = 1:numClusters
                distances(i, k) = sum((trainData(i, :) - centroids(k, :)).^2); % Euclidean distance
            end
        end
        [~, clusterIndices] = min(distances, [], 2);

        % Compute new centroids
        newCentroids = zeros(numClusters, size(trainData, 2));
        for k = 1:numClusters
            newCentroids(k, :) = mean(trainData(clusterIndices == k, :), 1);
        end

        % Check for convergence (if centroids do not change significantly)
        if norm(newCentroids - centroids) < tol
            disp('Convergence reached.');
            break;
        end
        centroids = newCentroids;
    end
end

% Training K-Means for multiple epochs
for epoch = 1:epochs
    % Train the K-Means model
    [centroids, clusterIndices] = trainKMeans(trainReducedFeatures, numClusters, maxIter, tol);

    % Evaluate clustering using Silhouette Score on training and validation sets
    trainSilhouetteScores(epoch) = mean(silhouette(trainReducedFeatures, clusterIndices));
    
    % Assign clusters to validation data
    distancesVal = zeros(size(valReducedFeatures, 1), numClusters);
    for i = 1:size(valReducedFeatures, 1)
        for k = 1:numClusters
            distancesVal(i, k) = sum((valReducedFeatures(i, :) - centroids(k, :)).^2);
        end
    end
    [valClusterIndices, ~] = min(distancesVal, [], 2);
    valSilhouetteScores(epoch) = mean(silhouette(valReducedFeatures, valClusterIndices));
    
    % Visualize clustering progress (plot centroids and data points)
    figure;
    hold on;
    gscatter(trainReducedFeatures(:, 1), trainReducedFeatures(:, 2), clusterIndices, 'rgb', 'xo', 8);
    plot(centroids(:, 1), centroids(:, 2), 'kx', 'MarkerSize', 10, 'LineWidth', 3);
    title(['Epoch ' num2str(epoch) ' - Clustering Progress']);
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    grid on;
    hold off;
end

% Plot the training Silhouette Score over epochs
figure;
subplot(1, 2, 1);
plot(1:epochs, trainSilhouetteScores, '-o', 'LineWidth', 2);
title('Training Silhouette Score Over Epochs');
xlabel('Epochs');
ylabel('Silhouette Score');
grid on;

% Plot the validation Silhouette Score over epochs
subplot(1, 2, 2);
plot(1:epochs, valSilhouetteScores, '-o', 'LineWidth', 2);
title('Validation Silhouette Score Over Epochs');
xlabel('Epochs');
ylabel('Silhouette Score');
grid on;

% 3. Final Clustering Evaluation on Test Set
distancesTest = zeros(size(testReducedFeatures, 1), numClusters);
for i = 1:size(testReducedFeatures, 1)
    for k = 1:numClusters
        distancesTest(i, k) = sum((testReducedFeatures(i, :) - centroids(k, :)).^2);
    end
end
[testClusterIndices, ~] = min(distancesTest, [], 2);

% Plot final clustering result on test set
figure;
gscatter(testReducedFeatures(:, 1), testReducedFeatures(:, 2), testClusterIndices, 'rgb', 'xo', 8);
title('Test Set Clustering');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
grid on;


%{

trainKMeans Function: This function trains the K-Means model, assigns clusters, updates centroids, and checks for convergence.
Silhouette Score: The Silhouette Score is calculated to evaluate the clustering quality on both the training and validation datasets.
Epoch Visualization: At each epoch, the centroids are updated and the clustering progress is visualized with gscatter to show how clusters evolve over time.
Training and Validation Silhouette Scores: The silhouette scores are tracked and plotted to show how well the clustering is performing during training.

%}