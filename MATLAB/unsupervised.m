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

% K-Means Clustering
numClusters = 3; % Number of clusters (adjust based on your data)
[clusterIndices, clusterCenters] = kmeans(combinedFeatures, numClusters);

% Visualize the clusters (you can reduce dimensions for visualization purposes)
% Reduce the features for visualization using PCA (optional)
[coeff, score, ~, ~, explained] = pca(combinedFeatures);

% Plot the first two principal components
figure;
gscatter(score(:, 1), score(:, 2), clusterIndices, 'rgb', 'xo', 8);
hold on;
plot(clusterCenters(:, 1), clusterCenters(:, 2), 'k*', 'MarkerSize', 10);
title('K-Means Clustering with PCA (Unsupervised Learning)');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster Centers', 'Location', 'Best');
hold off;

% Display the cluster assignments for each sample
disp('Cluster assignments for each sample:');
disp(clusterIndices);

% Optionally, display the centers of each cluster
disp('Cluster centers:');
disp(clusterCenters);
