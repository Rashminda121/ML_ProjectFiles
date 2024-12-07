
% clear data
clc;
clearvars;

% profile on

% Loading Data
folderPath = 'userfilesCombined';
fileList = dir(fullfile(folderPath, 'U*_Acc_TimeD_FreqD_FDay.mat'));

% Initialize a cell array to store the data for each file
Temp_Acc_Data = cell(1, length(fileList));

for nc = 1:length(fileList)
    % Load each file
    filePath = fullfile(folderPath, fileList(nc).name);
    T_Acc_Data_FDay = load(filePath);
    
    % Extract the required data and store in Temp_Acc_Data
    Temp_Acc_Data{nc} = T_Acc_Data_FDay.Acc_TDFD_Feat_Vec(1:36, 1:131);
end


% Concatenate data from all users into a single variable

Temp_Acc_Data_TDFD = [];

% Concatenate each 36-by-43 matrix vertically
for nc = 1:length(Temp_Acc_Data)
    Temp_Acc_Data_TDFD = [Temp_Acc_Data_TDFD; Temp_Acc_Data{nc}];
end


% Labeling data for each user 

% Number of rows in the concatenated data
num_rows = size(Temp_Acc_Data_TDFD, 1);


% Initialize an index for labeling
labelIndex = 1;

% Loop through the data in blocks of 36 rows
for i = 1:36:num_rows
    % Determine the end row for the current block
    endRow = min(i + 35, num_rows);
    
    % Create a temporary label array (0s for all rows)
    Temp_Acc_Data_TDFD_Labels = zeros(num_rows, 1);
    
    % Label the first 36 rows in this block as 1
    Temp_Acc_Data_TDFD_Labels(i:endRow) = 1;
    
    % Store the labeled data in the corresponding Temp_Acc_Data_FD_U variable
    eval(['Temp_Acc_Data_TDFD_U' num2str(labelIndex) ' = [Temp_Acc_Data_TDFD, Temp_Acc_Data_TDFD_Labels];']);

    % getting user data count
    Acc_Data_TDFD_U = labelIndex;

    % Creating user label set for the current labelIndex
    eval(['Temp_Acc_Data_TDFD_UL' num2str(labelIndex) ' = Temp_Acc_Data_TDFD_Labels;']);
    
    % Increment the label index for the next block
    labelIndex = labelIndex + 1;
end

% Temp_Acc_Data_TDFD_U = temp user dataset created
% Temp_Acc_Data_TDFD_UL = temp user labels created



% Traning data 

% load data for each temp user dataset

u_num = 1; % change u_num value to select a user from 1:10
hidden_layers = [10 5]; % change hidden layers 
training_per = 0.65;   % 60% training
testing_per = 0.35;   % 40% testing
validation = 0.0;  % validation percentage
num_epochs = 10;  % change epochs 
learning_rate = 0.002;   % change learning rate
regularization = 0.15;

% param_goal = 1e-3; % change MSE


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
idx = randperm(numSamples); % Shuffle indices
trainIdx = idx(1:round(training_per * numSamples));
testIdx = idx(round(training_per * numSamples) + 1:end);

%corrMatrix = corr(features); % Compute correlation matrix
%heatmap(corrMatrix);          % Visualize correlation


trainData = featuresPCA(trainIdx, :);
trainLabels = labels(trainIdx);

testData = featuresPCA(testIdx, :);
testLabels = labels(testIdx);


% Create and train the neural network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net = feedforwardnet(hidden_layers);

% Set the training function to Levenberg-Marquardt
net.trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation

net.divideParam.trainRatio = training_per;
net.divideParam.testRatio = testing_per;
net.divideParam.valRatio = validation; % No validation data
net.trainParam.epochs = num_epochs;
net.trainParam.lr = learning_rate;

% Add regularization
net.performParam.regularization = regularization; % Regularization value


disp(['Default Learning Rate: ', num2str(net.trainParam.lr)]);

% Train the network 
[net, tr] = train(net, trainData', trainLabels'); 


% Make Predictions on the Testing Set
predictions = net(testData');  % Predict on the testing set
predictions = round(predictions);  % Round predictions to 0 or 1

% Evaluate the Model
% Compute accuracy on the training and test sets
trainPredictions = net(trainData');
trainPredictions = round(trainPredictions);

trainAccuracy = sum(trainPredictions' == trainLabels) / length(trainLabels);
testAccuracy = sum(predictions' == testLabels) / length(testLabels);


trainAccuracyper = trainAccuracy * 100;
testAccuracyper = testAccuracy * 100;


% Data for the bar plot
accuracies = [trainAccuracyper, testAccuracyper];
categories = {'Training Accuracy', 'Testing Accuracy'};

% Create the bar plot
figure;
b = bar(accuracies);  % Create the bar plot

% Set different colors for each bar
b.FaceColor = 'flat';
b.CData(1, :) = [0.2, 0.6, 0.8];  % Color for Training Accuracy
b.CData(2, :) = [0.8, 0.2, 0.2];  % Color for Testing Accuracy

% Set the x-axis labels and other plot properties
set(gca, 'XTickLabel', categories, 'XTick', 1:2);
ylabel('Accuracy (%)');
title('Model Accuracy Comparison');
ylim([90, 100]);

% Display the plot
grid on;


% Annotate the bars with their values
for i = 1:length(accuracies)
    text(i, accuracies(i) + 0.5, sprintf('%.2f%%', accuracies(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
end

grid on;


% Plot the Performance Graph
figure;
plotperform(tr);  % Plot the performance graph including training, validation, and test

% Get the best performance (minimum error) and its corresponding epoch
bestPerformance = min(tr.perf);  % Find the best (lowest) performance
bestEpoch = find(tr.perf == bestPerformance, 1);  % Find the epoch with the best performance

% Add a horizontal line showing the best performance
yline(bestPerformance, '--r', 'Best Performance');  % Add the horizontal line


% View values 

% Enable data cursor mode to show values on hover
dcm = datacursormode(gcf);
set(dcm, 'Enable', 'on');

% Customize the data tip to show custom information
set(dcm, 'UpdateFcn', @(obj, event) customDataTip(obj, event, bestPerformance, bestEpoch));

% Function to customize the data tip
function output_txt = customDataTip(~, event_obj, bestPerformance, bestEpoch)
    % Get the position of the data point
    pos = event_obj.Position;
    
    % If the data point is close to the best performance, show custom info
    if abs(pos(2) - bestPerformance) < 1e-15  % Adjust the tolerance as needed
        output_txt = {['Best Performance: ', num2str(bestPerformance)], ...
                      ['Epoch: ', num2str(bestEpoch)]};
    else
        output_txt = {['Epoch: ', num2str(round(pos(1)))], ...
                      ['Performance: ', num2str(pos(2))]};
    end
end

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


% Define custom colors for each class
colors = [0.2 0.6 0.2;   % Green for TN
          0.8 0.2 0.2;   % Red for FP
          0.2 0.2 0.8;   % Blue for FN
          0.8 0.8 0.2];  % Yellow for TP

% Create a figure
figure;

% Plot each bar separately for clarity
hold on;  % Hold on to combine multiple plots in the same figure
h1 = bar(1, cm(1), 'FaceColor', colors(1, :));  % TN
h2 = bar(2, cm(2), 'FaceColor', colors(2, :));  % FP
h3 = bar(3, cm(3), 'FaceColor', colors(3, :));  % FN
h4 = bar(4, cm(4), 'FaceColor', colors(4, :));  % TP
hold off;  % Release the hold after plotting all bars

% Set the title, axis labels, and ticks
title('Confusion Matrix (Bar Plot)');
xlabel('Classes');
ylabel('Frequency');
xticks(1:4);
xticklabels({'TN', 'FP', 'FN', 'TP'}); % Can be adjusted based on confusion matrix size
grid on;

% Add a legend explaining the colors
legend([h1, h2, h3, h4], {'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)', 'True Positives (TP)'}, 'Location', 'northeast');



% If you want to label each bar with the actual values:
for i = 1:numel(cm)
    text(i, cm(i), num2str(cm(i)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end

% heat map for confusion matrics
figure;
heatmap(cm, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted', 'YLabel', 'Actual', ...
        'CellLabelFormat', '%d');

% Initialize mean and variance data
meanData = zeros(length(fileList), size(Temp_Acc_Data{1}, 2));
varianceData = zeros(size(meanData));
clusteringData = cell(length(fileList), 1);

% Determine optimal number of clusters once for all users (assuming data consistency)
allUserData = cell2mat(cellfun(@(x) zscore(x), Temp_Acc_Data, 'UniformOutput', false));
optimalK = findOptimalClusters(allUserData, 1:5);

% Loop through each user dataset
figure;
for nc = 1:length(fileList)
    userData = Temp_Acc_Data{nc};
    meanData(nc, :) = mean(userData);
    varianceData(nc, :) = var(userData);

    % Normalize user data
    userDataNorm = zscore(userData);

    % Perform k-means clustering with optimalK clusters
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

% Plot overall summaries
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
% legend('Location', 'best');

hold off;



% figure for the clustering plot
figure;
colors = lines(length(fileList));

% Loop through each user to plot their centroids
for i = 1:length(fileList)
    centroids = clusteringData{i};
    scatter(centroids(:, 1), centroids(:, 2), 50, colors(i, :), 'filled', 'DisplayName', sprintf('User %d', i));
    hold on;
end

% Updated plot title and axis labels
title('Clustering of Users Based on Normalized Feature Data');
xlabel('Feature Dimension 1');
ylabel('Feature Dimension 2');
grid on;
legend('Location', 'bestoutside');

hold off;



% --- Function to determine the optimal number of clusters using the Elbow Method---
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

    % Find the optimal k using the second derivative
    [~, optimalIdx] = min(diff(diff(sumD)));
    optimalK = kRange(optimalIdx + 1);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Compute intra-class variance for each user
intraClassVariance = zeros(length(fileList), 1);
for nc = 1:length(fileList)
    userData = Temp_Acc_Data{nc}; % User data
    intraClassVariance(nc) = mean(var(userData, 0, 1)); % Average variance across features
end

% Generate a colormap with distinct colors for each user
numUsers = length(fileList);
colors = lines(numUsers); % Generate 'lines' colormap with distinct colors

% Create a figure for the bar plot
figure;
hold on; % Allow multiple bars to be plotted on the same figure

% Plot each user's intra-class variance as a separate bar
bars = gobjects(numUsers, 1); % Preallocate graphics object array
for nc = 1:numUsers
    bars(nc) = bar(nc, intraClassVariance(nc), 'FaceColor', colors(nc, :), 'DisplayName', sprintf('User %d', nc));
end

% Customize plot appearance
title('Intra-Class Variance for Each User');
xlabel('User');
ylabel('Intra-Class Variance');
grid on;

% Add legend for each user
% legend('Location', 'bestoutside');

% Adjust x-axis ticks to match user indices
xticks(1:numUsers);
xticklabels(arrayfun(@(x) sprintf('User %d', x), 1:numUsers, 'UniformOutput', false));

hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% inter-class variance
% Initialize mean vectors for each user
numFeatures = size(Temp_Acc_Data{1}, 2);
userMeans = zeros(numUsers, numFeatures);

% Compute the mean vector for each user
for nc = 1:numUsers
    userData = Temp_Acc_Data{nc};
    userMeans(nc, :) = mean(userData, 1); % Mean of each feature for the user
end

% Compute the overall mean vector across all users
overallMean = mean(userMeans, 1);

% Compute inter-class variance (variance of user means across features)
interClassVariance = mean(var(userMeans, 0, 1));

% Display the inter-class variance
fprintf('Inter-Class Variance: %.4f\n', interClassVariance);

% Plot the inter-class variance as a bar chart
figure;
bar(interClassVariance, 'FaceColor', [0.5, 0, 0.5]);  
title('Inter-Class Variance');
ylabel('Variance Value');
set(gca, 'XTickLabel', {'Inter-Class Variance'});
grid on;


%%%%
% Compute variance of each feature's mean across all users
userMeanVariance = var(userMeans, 0, 1); % Variance of means for each feature

% Plot a bar plot of the variances
figure;
bar(userMeanVariance, 'FaceColor', [0.4, 0.6, 0.8]);
title('Variance of User Means Across Features');
xlabel('Feature Index');
ylabel('Variance Value');
grid on;


%%%%
% Centering meanData before applying PCA
meanDataCentered = meanData - mean(meanData);

% Perform PCA on the centered mean data
[coeff, score, ~, ~, explained] = pca(meanDataCentered);

% Generate a colormap for distinct colors for each user
numUsers = length(fileList);
colors = lines(numUsers); % 'lines' colormap provides distinct colors

% 2D PCA Visualization with distinct colors for each user
figure;
hold on; % Allow multiple points to be plotted
for i = 1:numUsers
    scatter(score(i, 1), score(i, 2), 100, 'filled', 'MarkerFaceColor', colors(i, :), 'DisplayName', sprintf('User %d', i));
end

% Title and axis labels
title('PCA: Inter-Class Variance Visualization');
xlabel(sprintf('Principal Component 1 (%.2f%%)', explained(1)));
ylabel(sprintf('Principal Component 2 (%.2f%%)', explained(2)));

% Customize appearance
grid on;
xlim([min(score(:, 1)) - 1, max(score(:, 1)) + 1]);
ylim([min(score(:, 2)) - 1, max(score(:, 2)) + 1]);

% Add legend
legend('show', 'Location', 'bestoutside');



%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% ROC Curve and AUC
% actual predicted probabilities from the model

[X, Y, T, AUC] = perfcurve(testLabels, predictions', 1);
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ', num2str(AUC), ')']);

disp(['AUC Value: ', num2str(AUC)]);
disp(' ');


% Learning Curves

figure;
plot(tr.epoch, tr.perf);
title('Training Performance vs Epoch');
xlabel('Epoch');
ylabel('Performance (Error)');



% Calculate Precision, Recall, and F1-Score

precision = cm(2,2) / (cm(2,2) + cm(1,2));  % TP / (TP + FP)
recall = cm(2,2) / (cm(2,2) + cm(2,1));     % TP / (TP + FN)
f1Score = 2 * (precision * recall) / (precision + recall);

% Display metrics
disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['F1-Score: ', num2str(f1Score)]);
disp(' ');

% Plot Precision, Recall, and F1-Score
metrics = [precision, recall, f1Score];
metricNames = {'Precision', 'Recall', 'F1-Score'};

figure;
b = bar(metrics);  % Create a bar plot

% Set different colors for each bar
b.FaceColor = 'flat';
b.CData(1, :) = [0.2, 0.6, 0.2];  % Green for Precision
b.CData(2, :) = [0.8, 0.2, 0.2];  % Red for Recall
b.CData(3, :) = [0.2, 0.2, 0.8];  % Blue for F1-Score

% Set the x-axis labels and other plot properties
set(gca, 'xticklabel', metricNames);  % Set the x-axis labels as metric names
ylabel('Score');  % Label for the y-axis
title('Model Evaluation Metrics');  % Title for the plot

% Display the plot
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
