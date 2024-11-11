t=@(x) x+2;
t(8)

%nprtool
%nnstart
clc;
% Example dataset: 36 samples and 43 features
%data = rand(36, 43);  % Replace this with your actual dataset
data =Acc_FD_Feat_Vec;

% Create a table for easier handling
dataTable = array2table(data, 'VariableNames', strcat('Feature', string(1:43)));

% Define churn criteria based on feature analysis
% Example criteria (customize based on your analysis)
dataTable.Churn = (dataTable.Feature1 < 0.1) | (dataTable.Feature2 < 0.05); % Adjust as necessary

% Convert logical to double (0 or 1)
dataTable.Churn = double(dataTable.Churn);

% Display the resulting table
disp(dataTable);
