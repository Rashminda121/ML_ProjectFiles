
% clear data
clc;
clearvars;

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
hidden_layers = 5;   % Hidden layers 
training_per = 0.6;  % Training percentage
testing_per = 0.4;   % Testing percentage
num_epochs = 1000;   % Epochs 

% param_goal = 1e-3; % change MSE

datasetName = ['Temp_Acc_Data_TDFD_U', num2str(u_num)];
data = eval(datasetName);

disp(['Result dataset size: ', num2str(size(data))]);
disp('');

% Split the Data into Training and Testing
numSamples = size(data, 1); 
idx = randperm(numSamples);

% Separate features and labels
features = data(:, 1:end-1);
labels = data(:, end);

% Split into training and testing
trainIdx = idx(1:round(training_per * numSamples));
testIdx = idx(round(training_per * numSamples) + 1:end);

trainData = features(trainIdx, :);
trainLabels = labels(trainIdx);

testData = features(testIdx, :);
testLabels = labels(testIdx);


% Create and Train the Neural Network
net = feedforwardnet(hidden_layers);


net.divideParam.trainRatio = training_per;
net.divideParam.testRatio = testing_per ;
net.divideParam.valRatio = 0.0;
net.trainParam.epochs = num_epochs;
% net.trainParam.goal = 1e-5;

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


% Plot the Performance Graph
figure;
plotperform(tr);

bestPerformance = min(tr.perf);
bestEpoch = find(tr.perf == bestPerformance, 1);
yline(bestPerformance, '--r', 'Best Performance');


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
heatmap(cm, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted', 'YLabel', 'Actual');


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

