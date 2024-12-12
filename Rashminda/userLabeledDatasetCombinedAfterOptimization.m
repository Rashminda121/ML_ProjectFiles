
% clear data
clc;
clearvars;

% profile on

% Loading Data
folderPath = 'userfilesCombined';
fileList = dir(fullfile(folderPath, 'U*_Acc_TimeD_FreqD_FDay.mat'));

% Cell array to store the data for each file
Temp_Acc_Data = cell(1, length(fileList));

for nc = 1:length(fileList)
    filePath = fullfile(folderPath, fileList(nc).name);
    T_Acc_Data_FDay = load(filePath);
    
    Temp_Acc_Data{nc} = T_Acc_Data_FDay.Acc_TDFD_Feat_Vec(1:36, 1:131);
end

Temp_Acc_Data_TDFD = [];

for nc = 1:length(Temp_Acc_Data)
    Temp_Acc_Data_TDFD = [Temp_Acc_Data_TDFD; Temp_Acc_Data{nc}];
end


% Labeling data for each user 
num_rows = size(Temp_Acc_Data_TDFD, 1);

labelIndex = 1;

for i = 1:36:num_rows
    endRow = min(i + 35, num_rows);
    Temp_Acc_Data_TDFD_Labels = zeros(num_rows, 1);

    Temp_Acc_Data_TDFD_Labels(i:endRow) = 1;
    eval(['Temp_Acc_Data_TDFD_U' num2str(labelIndex) ' = [Temp_Acc_Data_TDFD, Temp_Acc_Data_TDFD_Labels];']);

    Acc_Data_TDFD_U = labelIndex;
    eval(['Temp_Acc_Data_TDFD_UL' num2str(labelIndex) ' = Temp_Acc_Data_TDFD_Labels;']);

    labelIndex = labelIndex + 1;
end



% Traning data 

u_num = 1; % Select a user from 1:10
hidden_layers = [10 5];  % Hidden layers 
training_per = 0.65;     % Training percentage 
testing_per = 0.35;      % Testing percentage
validation = 0.0;        % Validation percentage
num_epochs = 10;         % Epochs 
learning_rate = 0.002;   % Learning rate
regularization = 0.15;   % Regularization rate


datasetName = ['Temp_Acc_Data_TDFD_U', num2str(u_num)];
data = eval(datasetName);

disp(['Result dataset size: ', num2str(size(data))]);
disp('');

% Separate features and labels
features = data(:, 1:end-1);
labels = data(:, end);

% Standardize the features (z-score normalization)
features = zscore(features);

% Apply PCA
[coeff, score, latent, tsquared, explained] = pca(features);

% Select components explaining 95% of the variance
cumExplained = cumsum(explained);
numComponents = find(cumExplained >= 95, 1);
featuresPCA = score(:, 1:numComponents);


% Split data into training and testing
numSamples = size(featuresPCA, 1);
idx = randperm(numSamples);
trainIdx = idx(1:round(training_per * numSamples));
testIdx = idx(round(training_per * numSamples) + 1:end);

trainData = featuresPCA(trainIdx, :);
trainLabels = labels(trainIdx);

testData = featuresPCA(testIdx, :);
testLabels = labels(testIdx);


% Create and train the neural network
net = feedforwardnet(hidden_layers);

net.trainFcn = 'trainlm';

net.divideParam.trainRatio = training_per;
net.divideParam.testRatio = testing_per;
net.divideParam.valRatio = validation;
net.trainParam.epochs = num_epochs;
net.trainParam.lr = learning_rate;

% regularization
net.performParam.regularization = regularization;

disp(['Default Learning Rate: ', num2str(net.trainParam.lr)]);

% Train the network 
[net, tr] = train(net, trainData', trainLabels'); 


% Make Predictions on the Testing Set
predictions = net(testData');  
predictions = round(predictions);

% Evaluate the Model
% Compute accuracy on the training and test sets
trainPredictions = net(trainData');
trainPredictions = round(trainPredictions);

trainAccuracy = sum(trainPredictions' == trainLabels) / length(trainLabels);
testAccuracy = sum(predictions' == testLabels) / length(testLabels);


trainAccuracyper = trainAccuracy * 100;
testAccuracyper = testAccuracy * 100;


% Plot Training and Testing accuracy  
accuracies = [trainAccuracyper, testAccuracyper];
categories = {'Training Accuracy', 'Testing Accuracy'};

figure;
b = bar(accuracies);

b.FaceColor = 'flat';
b.CData(1, :) = [0.2, 0.6, 0.8]; 
b.CData(2, :) = [0.8, 0.2, 0.2];

set(gca, 'XTickLabel', categories, 'XTick', 1:2);
ylabel('Accuracy (%)');
title('Model Accuracy Comparison');
ylim([90, 100]);

grid on;

for i = 1:length(accuracies)
    text(i, accuracies(i) + 0.5, sprintf('%.2f%%', accuracies(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
end

grid on;

% Plot the Performance Graph
figure;
plotperform(tr);

bestPerformance = min(tr.perf);
bestEpoch = find(tr.perf == bestPerformance, 1);

yline(bestPerformance, '--r', 'Best Performance');

% Plot explained variance
figure;
pareto(explained);
title('Explained Variance by Principal Components');
xlabel('Principal Component');
ylabel('Variance Explained (%)');

figure;
plot(cumExplained, '-o');
title('Cumulative Explained Variance');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');


% Confusion Matrix
cm = confusionmat(testLabels, predictions');
disp('Confusion Matrix:');
disp(cm);
disp(' ');

colors = [0.2 0.6 0.2; 
          0.8 0.2 0.2;  
          0.2 0.2 0.8;  
          0.8 0.8 0.2]; 


% Plot Confusion Matrix
figure;

hold on;
h1 = bar(1, cm(1), 'FaceColor', colors(1, :));  % TN
h2 = bar(2, cm(2), 'FaceColor', colors(2, :));  % FP
h3 = bar(3, cm(3), 'FaceColor', colors(3, :));  % FN
h4 = bar(4, cm(4), 'FaceColor', colors(4, :));  % TP
hold off;

title('Confusion Matrix (Bar Plot)');
xlabel('Classes');
ylabel('Frequency');
xticks(1:4);
xticklabels({'TN', 'FP', 'FN', 'TP'});
grid on;

legend([h1, h2, h3, h4], {'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)', 'True Positives (TP)'}, 'Location', 'northeast');

for i = 1:numel(cm)
    text(i, cm(i), num2str(cm(i)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end

% Heat map for confusion matrics
figure;
heatmap(cm, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted', 'YLabel', 'Actual', ...
        'CellLabelFormat', '%d');


% Initialize mean and variance data
meanData = zeros(length(fileList), size(Temp_Acc_Data{1}, 2));
varianceData = zeros(size(meanData));
clusteringData = cell(length(fileList), 1);

allUserData = cell2mat(cellfun(@(x) zscore(x), Temp_Acc_Data, 'UniformOutput', false));
optimalK = findOptimalClusters(allUserData, 1:5);

figure;

for nc = 1:length(fileList)
    userData = Temp_Acc_Data{nc};
    meanData(nc, :) = mean(userData);
    varianceData(nc, :) = var(userData);

    userDataNorm = zscore(userData);

    [idx, centroids] = kmeans(userDataNorm, optimalK, 'MaxIter', 300, 'Replicates', 5);
    clusteringData{nc} = centroids;

    % Plot Mean
    subplot(3, 1, 1);
    plot(meanData(nc, :), 'DisplayName', ['User ', num2str(nc)]);
    hold on;
    title('Mean of Features');
    xlabel('Feature Index');
    ylabel('Mean Value');

    % Plot Variance
    subplot(3, 1, 2);
    plot(varianceData(nc, :), '--', 'DisplayName', ['User ', num2str(nc)]);
    hold on;
    title('Variance of Features');
    xlabel('Feature Index');
    ylabel('Variance Value');
end

% Calculate overall mean, variance, and standard deviation across all users
overallMean = mean(meanData, 1);
overallVariance = mean(varianceData, 1);
overallStdDev = std(meanData, 0, 1);


% Plot overall mean on the mean subplot
subplot(3, 1, 1);
plot(overallMean, 'k-', 'LineWidth', 1, 'DisplayName', 'Overall Mean');
legend('Location', 'best');
hold off;

% Plot overall variance on the variance subplot
subplot(3, 1, 2);
plot(overallVariance, 'k--', 'LineWidth', 1, 'DisplayName', 'Overall Variance');
legend('Location', 'best');
hold off;

% Plot the overall standard deviation across features
subplot(3, 1, 3);
plot(overallStdDev, 'Color', [0.5, 0, 0.5], 'LineWidth', 2, 'DisplayName', 'Overall Standard Deviation');
title('Standard Deviation of Feature Means Across Users');
xlabel('Feature Index');
ylabel('Standard Deviation');
grid on;
hold off;

% Plot the clustering 
figure;

colors = lines(length(fileList));

for i = 1:length(fileList)
    centroids = clusteringData{i};
    scatter(centroids(:, 1), centroids(:, 2), 50, colors(i, :), 'filled', 'DisplayName', sprintf('User %d', i));
    hold on;
end

title('Clustering of Users Based on Normalized Feature Data');
xlabel('Feature Dimension 1');
ylabel('Feature Dimension 2');
grid on;
legend('Location', 'bestoutside');
hold off;



% Optimal number of clusters using the Elbow Method
function optimalK = findOptimalClusters(data, kRange)
    sumD = zeros(length(kRange), 1);

    for kIdx = 1:length(kRange)
        k = kRange(kIdx);
        [~, ~, sumd] = kmeans(data, k, 'Replicates', 5, 'Display', 'off');
        sumD(kIdx) = sum(sumd);
    end

    % Plot the Elbow Curve for visualization
    figure;
    plot(kRange, sumD, '-o');
    title('Elbow Method for Optimal k');
    xlabel('Number of Clusters (k)');
    ylabel('Sum of Squared Distances');
    grid on;

    [~, optimalIdx] = min(diff(diff(sumD)));
    optimalK = kRange(optimalIdx + 1);
end



% Intra-class variance for each user
intraClassVariance = zeros(length(fileList), 1);

for nc = 1:length(fileList)
    userData = Temp_Acc_Data{nc};
    intraClassVariance(nc) = mean(var(userData, 0, 1));
end

numUsers = length(fileList);
colors = lines(numUsers);

% Plot Intra-class variance for each user
figure;
hold on;

bars = gobjects(numUsers, 1);
for nc = 1:numUsers
    bars(nc) = bar(nc, intraClassVariance(nc), 'FaceColor', colors(nc, :), 'DisplayName', sprintf('User %d', nc));
end

title('Intra-Class Variance for Each User');
xlabel('User');
ylabel('Intra-Class Variance');
grid on;

xticks(1:numUsers);
xticklabels(arrayfun(@(x) sprintf('User %d', x), 1:numUsers, 'UniformOutput', false));
hold off;



% Inter-class variance
numFeatures = size(Temp_Acc_Data{1}, 2);
userMeans = zeros(numUsers, numFeatures);

for nc = 1:numUsers
    userData = Temp_Acc_Data{nc};
    userMeans(nc, :) = mean(userData, 1);
end

overallMean = mean(userMeans, 1);

% Compute inter-class variance (variance of user means across features)
interClassVariance = mean(var(userMeans, 0, 1));
fprintf('Inter-Class Variance: %.4f\n', interClassVariance);


% Plot the inter-class variance
figure;
bar(interClassVariance, 'FaceColor', [0.5, 0, 0.5]);  
title('Inter-Class Variance');
ylabel('Variance Value');
set(gca, 'XTickLabel', {'Inter-Class Variance'});
grid on;


% Inter-class variance per user
interClassVariancePerUser = sum((userMeans - overallMean).^2, 2) / numFeatures;

% Plot the inter-class variance for each user
figure;
colors = lines(numUsers);

barHandle = bar(interClassVariancePerUser);
barHandle.FaceColor = 'flat'; 

for i = 1:numUsers
    barHandle.CData(i, :) = colors(i, :);
end

title('Inter-Class Variance for Each User');
xlabel('Users');
ylabel('Variance Value');
grid on;

userLabels = arrayfun(@(x) sprintf('User %d', x), 1:numUsers, 'UniformOutput', false);
set(gca, 'XTick', 1:numUsers, 'XTickLabel', userLabels);


% Variance of each feature's mean across all users
userMeanVariance = var(userMeans, 0, 1);

% Plot the variances
figure;
bar(userMeanVariance, 'FaceColor', [0.4, 0.6, 0.8]);
title('Variance of User Means Across Features');
xlabel('Feature Index');
ylabel('Variance Value');
grid on;


% PCA on the centered mean data
meanDataCentered = meanData - mean(meanData);
[coeff, score, ~, ~, explained] = pca(meanDataCentered);

numUsers = length(fileList);
colors = lines(numUsers);

% Plot PCA Visualization
figure;
hold on;
for i = 1:numUsers
    scatter(score(i, 1), score(i, 2), 100, 'filled', 'MarkerFaceColor', colors(i, :), 'DisplayName', sprintf('User %d', i));
end

title('PCA: Inter-Class Variance Visualization');
xlabel(sprintf('Principal Component 1 (%.2f%%)', explained(1)));
ylabel(sprintf('Principal Component 2 (%.2f%%)', explained(2)));

grid on;
xlim([min(score(:, 1)) - 1, max(score(:, 1)) + 1]);
ylim([min(score(:, 2)) - 1, max(score(:, 2)) + 1]);
legend('show', 'Location', 'bestoutside');



% ROC Curve and AUC
[X, Y, T, AUC] = perfcurve(testLabels, predictions', 1);
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);

disp(['AUC Value: ', num2str(AUC)]);
disp(' ');


% Plot Learning Curves
figure;
plot(tr.epoch, tr.perf);
title('Training Performance vs Epoch');
xlabel('Epoch');
ylabel('Performance (Error)');


% Calculate Precision, Recall, and F1-Score
precision = cm(2,2) / (cm(2,2) + cm(1,2));
recall = cm(2,2) / (cm(2,2) + cm(2,1));
f1Score = 2 * (precision * recall) / (precision + recall);

disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['F1-Score: ', num2str(f1Score)]);
disp(' ');

% Plot Precision, Recall, and F1-Score
metrics = [precision, recall, f1Score];
metricNames = {'Precision', 'Recall', 'F1-Score'};

figure;
b = bar(metrics);

b.FaceColor = 'flat';
b.CData(1, :) = [0.2, 0.6, 0.2]; 
b.CData(2, :) = [0.8, 0.2, 0.2]; 
b.CData(3, :) = [0.2, 0.2, 0.8];  

set(gca, 'xticklabel', metricNames);
ylabel('Score');
title('Model Evaluation Metrics');
grid on;

% Display the best training performance and epoch
disp(['Best Training Performance: ', num2str(bestPerformance)]);
disp(['Epoch of Best Performance: ', num2str(bestEpoch)]);
disp(' ');

% Display overall accuracy
disp(['Training Accuracy: ', sprintf('%.2f', trainAccuracy * 100), '%']);
disp(['Testing Accuracy: ', sprintf('%.2f',testAccuracy * 100), '%']);
disp(' ');

% profile off
% profile viewer;
% profile clear;
