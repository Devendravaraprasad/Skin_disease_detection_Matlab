classdef final < matlab.apps.AppBase

    properties (Access = private)
        UIFigure
        UploadImageButton
        DetectImageButton
        ResetButton
        ExitButton
        Axes
        myNet
        classLabels
    end

    methods (Access = private)

        function createComponents(app)
            app.UIFigure = uifigure('Name', 'Skin Disease Detection', 'Position', [100, 100, 640, 480]);
            app.Axes = uiaxes(app.UIFigure, 'Position', [10, 50, 620, 400]);

            app.UploadImageButton = uibutton(app.UIFigure, 'Text', 'Upload Image', 'Position', [20, 10, 100, 22], 'ButtonPushedFcn', @(src, event) uploadImage(app, src, event));
            app.DetectImageButton = uibutton(app.UIFigure, 'Text', 'Detect Disease', 'Position', [140, 10, 100, 22], 'ButtonPushedFcn', @(src, event) detectDisease(app, src, event));
            app.ResetButton = uibutton(app.UIFigure, 'Text', 'Reset', 'Position', [260, 10, 60, 22], 'ButtonPushedFcn', @(src, event) resetApp(app, src, event));
            app.ExitButton = uibutton(app.UIFigure, 'Text', 'Exit', 'Position', [330, 10, 60, 22], 'ButtonPushedFcn', @(src, event) exitApp(app, src, event));

            load myNet1;
            app.myNet = myNet1;

            % Load class labels
            load classLabels.mat;
            app.classLabels = classLabels;
        end
    end

    methods (Access = private)

        function uploadImage(app, ~, ~)
            [filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png'}, 'Select an Image File');
            if isequal(filename, 0) || isequal(pathname, 0)
                return;  % User canceled the operation
            end

            imagePath = fullfile(pathname, filename);
            img = imread(imagePath);

            % Display the uploaded image
            imshow(img, 'Parent', app.Axes);
        end

  function detectDisease(app, ~, ~)
    % Check if an image has been uploaded
    if isempty(app.Axes.Children)
        errordlg('Please upload an image first.', 'Error');
        return;
    end

    % Get the uploaded image from the Axes
    img = app.Axes.Children.CData;

    % Resize the image to match the input size expected by the neural network
    img = imresize(img, [227, 227]);

    % Classify the resized image using the pre-trained neural network
    [label, scores] = classify(app.myNet, img);

    % Display the result
    title(app.Axes, char(label));

    % Display the confidence scores
    disp(['Predicted Label: ', char(label)]);
    disp(['Confidence Scores: ', num2str(scores)]);
end


        function resetApp(app, ~, ~)
            % Reset the UIAxes
            cla(app.Axes);

            % Reset the title
            title(app.Axes, '');
        end

        function exitApp(app, ~, ~)
            % Close the UIFigure
            delete(app.UIFigure);
        end
    end

    methods (Access = public)

        function app = final()
            createComponents(app);
            app.UIFigure.Visible = 'on';
        end
    end
end