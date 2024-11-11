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

% 1. Manual PCA Implementation (No Toolbox)
% Step 1: Compute the covariance matrix
covMatrix = cov(combinedFeatures);

% Step 2: Compute eigenvectors and eigenvalues using eig function
[eigenvectors, eigenvalues] = eig(covMatrix);

% Step 3: Sort eigenvalues and eigenvectors in descending order
[sortedEigenvalues, sortIdx] = sort(diag(eigenvalues), 'descend');
sortedEigenvectors = eigenvectors(:, sortIdx);

% Step 4: Select top 'numComponents' eigenvectors (Principal Components)
numComponents = 3; % Choose the number of principal components
reducedFeatures = combinedFeatures * sortedEigenvectors(:, 1:numComponents);

% 2. K-Means Clustering (No Toolbox)
numClusters = 3;  % Number of clusters
maxIter = 100;    % Maximum iterations
tol = 1e-4;       % Convergence tolerance

% Initialize centroids randomly
centroids = reducedFeatures(randperm(size(reducedFeatures, 1), numClusters), :);

% K-Means Iteration
for iter = 1:maxIter
    % Assign samples to the nearest centroid using Euclidean distance
    distances = zeros(size(reducedFeatures, 1), numClusters);
    for i = 1:size(reducedFeatures, 1)
        for k = 1:numClusters
            distances(i, k) = sum((reducedFeatures(i, :) - centroids(k, :)).^2); % Euclidean distance
        end
    end
    [~, clusterIndices] = min(distances, [], 2);
    
    % Compute new centroids
    newCentroids = zeros(numClusters, size(reducedFeatures, 2));
    for k = 1:numClusters
        newCentroids(k, :) = mean(reducedFeatures(clusterIndices == k, :), 1);
    end
    
    % Check for convergence (if centroids do not change significantly)
    if norm(newCentroids - centroids) < tol
        disp('Convergence reached.');
        break;
    end
    centroids = newCentroids;
end

% 3. Plot the clusters in PCA space (2D)
figure;

% Plot each cluster with a different color
hold on;
colors = ['r', 'g', 'b']; % Colors for clusters
for k = 1:numClusters
    clusterData = reducedFeatures(clusterIndices == k, :);
    scatter(clusterData(:, 1), clusterData(:, 2), 50, colors(k), 'filled');
end
hold off;

title('K-Means Clustering with PCA (Unsupervised Learning)');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend({'Cluster 1', 'Cluster 2', 'Cluster 3'}, 'Location', 'Best');



%The plot will show your data points (each data point corresponds to a user with 
% combined features) in a reduced 2D space, where each point is assigned to one of the 
% clusters using K-Means clustering. Each cluster will have a different color.