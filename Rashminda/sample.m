
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
    
    % Increment the label index for the next block
    labelIndex = labelIndex + 1;
end




% Traning data 

% Split the Data into Training (60%) and Testing (40%)
numSamples = size(Temp_Acc_Data_FD_U1, 1);
idx = randperm(numSamples);  % Random permutation of indices for data shuffling

% Separate features and labels
features = Temp_Acc_Data_FD_U1(:, 1:end-1);  % All columns except the last one are features
labels = Temp_Acc_Data_FD_U1(:, end);        % Last column is the label

% Split into training (60%) and testing (40%)
trainIdx = idx(1:round(0.6 * numSamples));
testIdx = idx(round(0.6 * numSamples) + 1:end);

trainData = features(trainIdx, :);
trainLabels = labels(trainIdx);

testData = features(testIdx, :);
testLabels = labels(testIdx);

% Create and Train the Neural Network
% Use a feedforward network with hidden layers [4 2 4]
net = feedforwardnet([4 2 4]);  % Hidden layers with 4, 2, and 4 neurons

% Configure the network division (training, validation, and testing)
net.divideParam.trainRatio = 0.6; % 60% training
% net.divideParam.valRatio = 0.2;   % 20% validation (used for performance tracking)
net.divideParam.testRatio = 0.4;  % 40% testing (used for performance tracking)
net.divideParam.valRatio = 0.0;  % Set validation ratio to 0 (no validation data used)

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


figure;
heatmap(cm, 'Title', 'Confusion Matrix', 'XLabel', 'Predicted', 'YLabel', 'Actual');



% Display the best training performance and epoch
disp(['Best Training Performance: ', num2str(bestPerformance)]);
disp(['Epoch of Best Performance: ', num2str(bestEpoch)]);

% Display overall accuracy
disp(['Training Accuracy: ', sprintf('%.2f', trainAccuracy * 100), '%']);
disp(['Testing Accuracy: ', sprintf('%.2f',testAccuracy * 100), '%']);


