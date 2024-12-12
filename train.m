clc
clear all
close all
warning off

% Load pre-trained AlexNet
g = alexnet;
layers = g.Layers;

% Specify the full path to your dataset
datasetPath = 'C:\Users\deven\OneDrive\Desktop\BNMIT\3rd year\5TH sem\Skin disease\cr';

% Create an imageDatastore for the dataset
allImages = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Save the subfolder names
classLabels = allImages.Labels;

% Create subfolders for diseases
uniqueDiseases = unique(classLabels);
uniqueDiseasesCell = cellstr(uniqueDiseases);  % Convert to cell array
for i = 1:numel(uniqueDiseasesCell)
    diseaseFolder = fullfile(datasetPath, uniqueDiseasesCell{i});
    mkdir(diseaseFolder);
end

% Copy images to their respective disease subfolders
reset(allImages);
while hasdata(allImages)
    [img, info] = read(allImages);
    
    % Get the label for the current image
    currentLabel = info.Label;
    
    % Convert currentLabel to string explicitly
    diseaseFolder = fullfile(datasetPath, char(currentLabel));
    
    % Get the file path directly from the Files property
    [~, fileName, fileExt] = fileparts(info.Filename);
    imageName = strcat(fileName, fileExt);
    
    imwrite(img, fullfile(diseaseFolder, imageName));
end

% Resize images to match the input size expected by AlexNet
inputSize = [227, 227, 3];
augmentedImages = augmentedImageDatastore(inputSize(1:2), allImages, ...
    'ColorPreprocessing', 'gray2rgb', ...
    'DataAugmentation', imageDataAugmenter('RandXReflection', true, 'RandYReflection', true, 'RandRotation', [-20, 20]));

% Define the number of classes
numClasses = numel(unique(classLabels));

% Create a new AlexNet with the modified layers
newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newClassificationLayer = classificationLayer('Name', 'new_output');
layersToReplace = [
    layers(1:end-3)
    newFcLayer
    newClassificationLayer
];

% Define fine-tuning options
fineTuneOpts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.0001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'Plots', 'training-progress');

% Fine-tune the network
myNet = trainNetwork(augmentedImages, layersToReplace, fineTuneOpts);

% Save the subfolder names and the fine-tuned network to MAT files
save('classLabels.mat', 'classLabels');
save('myNet.mat', 'myNet');