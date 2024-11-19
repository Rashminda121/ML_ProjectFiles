
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


% user 1
% Concatenate data from all users into a single variable

Temp_Acc_Data_FD = [];

for nc = 1:length(Temp_Acc_Data)
    % Concatenate each 36-by-43 matrix vertically
    Temp_Acc_Data_FD = [Temp_Acc_Data_FD; Temp_Acc_Data{nc}];
end


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