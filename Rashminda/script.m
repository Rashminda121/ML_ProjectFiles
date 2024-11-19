
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
