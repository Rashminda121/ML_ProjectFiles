% Clear data
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
for nc = 1:length(Temp_Acc_Data)
    Temp_Acc_Data_FD = [Temp_Acc_Data_FD; Temp_Acc_Data{nc}];
end

% Labeling data for each user
num_rows = size(Temp_Acc_Data_FD, 1);
labelIndex = 1;
for i = 1:36:num_rows
    endRow = min(i + 35, num_rows);
    Temp_Acc_Data_FD_Labels = zeros(num_rows, 1);
    Temp_Acc_Data_FD_Labels(i:endRow) = 1;
    
    eval(['Temp_Acc_Data_FD_U' num2str(labelIndex) ' = [Temp_Acc_Data_FD, Temp_Acc_Data_FD_Labels];']);
    eval(['Temp_Acc_Data_FD_UL' num2str(labelIndex) ' = Temp_Acc_Data_FD_Labels;']);
    labelIndex = labelIndex + 1;
end

% Training data
u_num = 1;  % Change u_num value to select a user from 1:10
hidden_layers = [4 2 4];  % Change hidden layers 
training_per = 0.6;  % 60% training
testing_per = 0.4;  % 40% testing
num_epochs = 1000;  % Change epochs 
learning_rate = 0.01;  % Change learning rate
param_goal = 1e-3;  % Change MSE

datasetName = ['Temp_Acc_Data_FD_U', num2str(u_num)];
data = eval(datasetName);

% Split the Data into Training (60%) and Testing (40%)
numSamples = size(data, 1);
idx = randperm(numSamples);  % Random permutation of indices for data shuffling

% Separate features and labels
features = data(:, 1:end-1);  % All columns except the last one are features
labels = data(:, end);  % Last column is the label

% Split into training (60%) and testing (40%)
trainIdx = idx(1:round(training_per * numSamples));
testIdx = idx(round(training_per * numSamples) + 1:end);

trainData = features(trainIdx, :);
trainLabels = labels(trainIdx);

testData = features(testIdx, :);
testLabels = labels(testIdx);

% Create and Train the Neural Network
net = feedforwardnet(hidden_layers);  % Hidden layers with 4, 2, and 4 neurons

% Configure the network division (training, validation, and testing)
net.divideParam.trainRatio = training_per;  % 60% training
net.divideParam.testRatio = testing_per;  % 40% testing
net.divideParam.valRatio = 0.0;  % Set validation ratio to 0 (no validation data used)

% Configure Training Parameters (Optional)
net.trainParam.epochs = num_epochs;  % Maximum epochs
net.trainParam.lr = learning_rate;  % Learning rate

% Train the network (with validation and testing)
[net, tr] = train(net, trainData', trainLabels');  % Note the transpose (') for correct input format

% Make Predictions on the Testing Set
predictions = net(testData');  % Predict on the testing set
predictions = round(predictions);  % Round predictions to 0 or 1

% Evaluate the Model
trainPredictions = net(trainData');
trainPredictions = round(trainPredictions);

trainAccuracy = sum(trainPredictions' == trainLabels) / length(trainLabels);
testAccuracy = sum(predictions' == testLabels) / length(testLabels);

% Display the results
disp(['Training Accuracy: ', num2str(trainAccuracy * 100), '%']);
disp(['Testing Accuracy: ', num2str(testAccuracy * 100), '%']);

% Plot the Performance Graph
figure;
plotperform(tr);  % Plot the performance graph including training, validation, and test

% Get the best performance (minimum error) and its corresponding epoch
bestPerformance = min(tr.perf);  % Find the best (lowest) performance
bestEpoch = find(tr.perf == bestPerformance, 1);  % Find the epoch with the best performance

% Add a horizontal line showing the best performance
yline(bestPerformance, '--r', 'Best Performance');  % Add the horizontal line

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

%%

% New Section: Mean, Variance, and Clustering Analysis
meanData = zeros(length(fileList), size(Temp_Acc_Data{1}, 2));
varianceData = zeros(size(meanData));
clusteringData = zeros(length(fileList), 2); 

figure;
for nc = 1:length(fileList)
    userData = Temp_Acc_Data{nc};
    meanData(nc, :) = mean(userData);
    varianceData(nc, :) = var(userData);
    
    % Perform clustering
    [~, centroids] = kmeans(userData, 2);
    clusteringData(nc, :) = centroids(:, 1)';  % Assuming 2 centroids with x-coordinates
    
    % Plot Mean
    subplot(3, 1, 1);
    plot(meanData(nc, :), 'DisplayName', ['User ', num2str(nc)]);
    hold on;
    title('Mean of Features');
    xlabel('Feature Index');
    ylabel('Mean Value');
    legend;

    % Plot Variance
    subplot(3, 1, 2);
    plot(varianceData(nc, :), '--', 'DisplayName', ['User ', num2str(nc)]);
    hold on;
    title('Variance of Features');
    xlabel('Feature Index');
    ylabel('Variance Value');
    legend;
end

% Clustering Plot
subplot(3, 1, 3);
hold on; % Allow multiple plots on the same axes

% Generate a unique color for each user
numUsers = size(clusteringData, 1); % Number of users
colors = lines(numUsers); % Generate a colormap with distinct colors

% Plot each user's data point with a unique color
for i = 1:numUsers
    scatter(clusteringData(i, 1), clusteringData(i, 2), 50, colors(i, :), 'filled', 'DisplayName', sprintf('User %d', i));
end

title('User Clustering');
xlabel('Centroid 1');
ylabel('Centroid 2');
grid on;

% Add legend
legend('Location', 'bestoutside');

hold off; % Release hold on the current axes



