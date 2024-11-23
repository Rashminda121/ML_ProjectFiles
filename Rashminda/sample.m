
% clear data
clc;
clearvars;

% Loading Data
folderPath = 'userfilesFrq';
fileList = dir(fullfile(folderPath, 'U*_Acc_FreqD_FDay.mat'));

% Initialize a cell array to store the data for each file
Temp_Acc_Data = cell(1, length(fileList));

for nc = 1:length(fileList)
    % Load each file
    filePath = fullfile(folderPath, fileList(nc).name);
    T_Acc_Data_FD_Day = load(filePath);
    
    % Extract the required data and store in Temp_Acc_Data
    Temp_Acc_Data{nc} = T_Acc_Data_FD_Day.Acc_FD_Feat_Vec(1:36, 1:43);
end


% Concatenate data from all users into a single variable

Temp_Acc_Data_FD = [];

% Concatenate each 36-by-43 matrix vertically
for nc = 1:length(Temp_Acc_Data)
    Temp_Acc_Data_FD = [Temp_Acc_Data_FD; Temp_Acc_Data{nc}];
end



% Labeling data for each user 

% Number of rows in the concatenated data
num_rows = size(Temp_Acc_Data_FD, 1);

% Initialize an index for labeling
labelIndex = 1;

% Loop through the data in blocks of 36 rows
for i = 1:36:num_rows
    % Determine the end row for the current block
    endRow = min(i + 35, num_rows);
    
    % Create a temporary label array (0s for all rows)
    Temp_Acc_Data_FD_Labels = zeros(num_rows, 1);
    
    % Label the first 36 rows in this block as 1
    Temp_Acc_Data_FD_Labels(i:endRow) = 1;
    
    % Store the labeled data in the corresponding Temp_Acc_Data_FD_U variable
    eval(['Temp_Acc_Data_FD_U' num2str(labelIndex) ' = [Temp_Acc_Data_FD, Temp_Acc_Data_FD_Labels];']);

    % getting user data count
    Acc_Data_FD_U = labelIndex;

    % Creating user label set for the current labelIndex
    eval(['Temp_Acc_Data_FD_UL' num2str(labelIndex) ' = Temp_Acc_Data_FD_Labels;']);
    
    % Increment the label index for the next block
    labelIndex = labelIndex + 1;
end

% Temp_Acc_Data_FD_U = temp user dataset created
% Temp_Acc_Data_FD_UL = temp user labels created



% Traning data 

% load data for each temp user dataset

u_num = 1; % change u_num value to select a user from 1:10
hidden_layers = [4 2 4]; % change hidden layers 
training_per = 0.6;   % 60% training
testing_per = 0.4;   % 40% testing
num_epochs = 1000;  % change epochs 
learning_rate = 0.01;   % change learning rate
param_goal = 1e-3; % change MSE


datasetName = ['Temp_Acc_Data_FD_U', num2str(u_num)];
data = eval(datasetName);

% Split the Data into Training (60%) and Testing (40%)
numSamples = size(data, 1); 
idx = randperm(numSamples);  % Random permutation of indices for data shuffling

% Separate features and labels
features = data(:, 1:end-1);  % All columns except the last one are features
labels = data(:, end);        % Last column is the label

% Split into training (60%) and testing (40%)
trainIdx = idx(1:round(training_per * numSamples));
testIdx = idx(round(training_per * numSamples) + 1:end);

trainData = features(trainIdx, :);
trainLabels = labels(trainIdx);

testData = features(testIdx, :);
testLabels = labels(testIdx);


% Create and Train the Neural Network

% Use a feedforward network with hidden layers [4 2 4]
net = feedforwardnet(hidden_layers);  % Hidden layers with 4, 2, and 4 neurons


% Configure the network division (training, validation, and testing)
net.divideParam.trainRatio = training_per; % 60% training
% net.divideParam.valRatio = 0.2;   % 20% validation (used for performance tracking)
net.divideParam.testRatio = testing_per ;  % 40% testing (used for performance tracking)
net.divideParam.valRatio = 0.0;  % Set validation ratio to 0 (no validation data used)


% Configure Training Parameters (Optional)
net.trainParam.epochs = num_epochs;        % Maximum epochs
% net.trainParam.goal = 1e-5;          % Performance goal
net.trainParam.lr = learning_rate;    % Learning rate

% Set the Performance Goal
% net.trainParam.goal = param_goal;  % Stop training when MSE is less than 1e-5

% net.trainParam.min_grad = 1e-6;  % Minimum gradient


disp(['Default Learning Rate: ', num2str(net.trainParam.lr)]);

% Train the network (with validation and testing)
[net, tr] = train(net, trainData', trainLabels');  % Note the transpose (') for correct input format

% Make Predictions on the Testing Set
predictions = net(testData');  % Predict on the testing set
predictions = round(predictions);  % Round predictions to 0 or 1

% Evaluate the Model
% Compute accuracy on the training and test sets
trainPredictions = net(trainData');
trainPredictions = round(trainPredictions);

trainAccuracy = sum(trainPredictions' == trainLabels) / length(trainLabels);
testAccuracy = sum(predictions' == testLabels) / length(testLabels);

%{ 

Display the results
disp('Training Accuracy:');
disp(trainAccuracy * 100);

disp('Testing Accuracy:');
disp(testAccuracy * 100); 

%}

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


% Confusion Matrix
cm = confusionmat(testLabels, predictions');
disp('Confusion Matrix:');
disp(cm);
disp(' ');

% Create a bar plot for the confusion matrix
figure;
bar(cm(:), 'FaceColor', [0.2 0.6 0.2]);
title('Confusion Matrix (Bar Plot)');
xlabel('Classes');
ylabel('Frequency');
xticks(1:numel(cm));
xticklabels({'TN', 'FP', 'FN', 'TP'}); % can be adjust based on confusion matrix size
grid on;

% If you want to label each bar with the actual values:
for i = 1:numel(cm)
    text(i, cm(i), num2str(cm(i)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end

% heat map for confusion matrics
figure;
heatmap(cm, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted', 'YLabel', 'Actual');


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
bar(metrics);  % Create a bar plot
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


% add gradient and mu values