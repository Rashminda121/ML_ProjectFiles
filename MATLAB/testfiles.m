
clc;
clearvars;

% Define the folder path containing .mat files
folderPath = 'mlproject/u1';

% Get a list of all .mat files in the folder
matFiles = dir(fullfile(folderPath, '*.mat'));

% Check if there are any .mat files in the folder
if isempty(matFiles)
    error('No .mat files found in the specified folder.');
end

% Loop through each .mat file and load its variables
for k = 1:length(matFiles)
    % Get the full path of the .mat file
    fileName = matFiles(k).name;
    fullFilePath = fullfile(folderPath, fileName);
    
    % Display the name of the file being loaded
    fprintf('Loading %s...\n', fileName);
    
    % Load variables from the .mat file
    data = load(fullFilePath);
    
    % Access variables in the loaded data (each variable can be accessed by name)
    variableNames = fieldnames(data);
    for i = 1:length(variableNames)
        variableName = variableNames{i};
        assignin('base', variableName, data.(variableName));  % Save each variable to the base workspace
    end
end

disp('All .mat files have been loaded successfully.');
