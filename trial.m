%% Nigerian Currency Detection System - Robust Version
% Features:
% - Fixed global variable handling
% - Improved SVM training with proper cross-validation
% - Enhanced error handling
% - Better feature extraction

% =====================
% Step 1: Start
% =====================

clc; clear; close all;

%% Step 2: Configuration
% (Set parameters for training and preprocessing)
numEpochs = 9;
validationSplit = 0.15;
dataAugmentation = true;
standard_size = [256, 256];

%% Step 3: Initialize Global Variables
% (Prepare variables for models and normalization)
global mu sigma final_svm_model cnn_model knn_model
mu = [];
sigma = [];
final_svm_model = [];
cnn_model = [];
knn_model = [];

%% Step 4: Load Dataset
% (Load images and labels from dataset folders)
disp('Loading data...');
try
    [fake_detection_features, denomination_data, labels_fake, labels_denomination] = loadData(standard_size);
catch ME
    error('Data loading failed: %s', ME.message);
end

%% Step 5: Preprocessing (Grayscale + Resize)
% (Handled inside loadData and extractCurrencyFeatures)

%% Step 6: Verify Dataset Quality
if length(unique(labels_fake)) < 2
    error('Dataset must contain both genuine and fake samples');
end

disp(['Genuine samples: ' num2str(sum(labels_fake==1))]);
disp(['Fake samples: ' num2str(sum(labels_fake==2))]);

%% Step 7: Segmentation (Threshold + ROI)
% (Segmentation is performed in createMask and showResults)

%% Step 8: Classification (SVM)
% (Train SVM classifier for fake/genuine detection)
disp('Training SVM...');
[X_train, X_test, y_train, y_test] = prepareData(fake_detection_features, labels_fake, validationSplit);

% Train with proper hyperparameter optimization
final_svm_model = trainSVM(X_train, y_train);

% Evaluate
[y_pred, scores] = predict(final_svm_model, X_test);
accuracy = sum(y_pred == y_test)/length(y_test)*100;
disp(['Test accuracy: ' num2str(accuracy) '%']);

%% Step 9: Classification (Denomination - CNN/KNN)
% (Train denomination classifier using CNN or KNN)
disp('Training denomination classifier...');
if license('test','neural_network_toolbox') && ~isempty(ver('nnet'))
    cnn_model = trainCNN(denomination_data, labels_denomination, standard_size, numEpochs, dataAugmentation);
else
    knn_model = trainKNN(denomination_data, labels_denomination, validationSplit);
end

%% Step 10: Save Models
% (Save trained models and normalization parameters)
saveModels(final_svm_model, cnn_model, knn_model, mu, sigma);

%% Step 11: Output (Real or Fake)
% (Launch UI for user to test images and see results)
runCurrencyDetection();

% =====================
% Step 12: End
% =====================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Supporting Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [features, img_data, labels_fake, labels_denom] = loadData(standard_size)
    features = [];
    img_data = {};
    labels_fake = [];
    labels_denom = [];
    
    if ~exist('train', 'dir')
        error('Dataset directory not found');
    end
    
    % Process genuine notes
    processClass('genuine', 1, standard_size);
    
    % Process fake notes
    processClass('fake', 2, standard_size);
    
    function processClass(class_dir, class_label, img_size)
        denominations = dir(fullfile('train', class_dir, '*'));
        denominations = denominations(~ismember({denominations.name}, {'.', '..'}));
        
        for i = 1:length(denominations)
            img_files = dir(fullfile('train', class_dir, denominations(i).name, '*.jpg'));
            
            for j = 1:length(img_files)
                img_path = fullfile('train', class_dir, denominations(i).name, img_files(j).name);
                try
                    img = imresize(imread(img_path), img_size);
                    feat = extractCurrencyFeatures(img);
                    
                    features = [features; feat];
                    img_data{end+1} = img;
                    labels_fake = [labels_fake; class_label];
                    labels_denom = [labels_denom; str2double(denominations(i).name)];
                catch ME
                    warning('Failed to process %s: %s', img_path, ME.message);
                end
            end
        end
    end
end

function svm_model = trainSVM(X_train, y_train)
    % Optimize hyperparameters properly
    rng(1); % For reproducibility
    svm_model = fitcsvm(X_train, y_train, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
        'HyperparameterOptimizationOptions', struct(...
            'AcquisitionFunctionName', 'expected-improvement-plus', ...
            'ShowPlots', false, ...
            'KFold', 5));
    
    % Ensure we can get probabilities
    try
        svm_model = fitPosterior(svm_model);
    catch
        warning('Could not fit posterior probabilities');
    end
end

function [X_train, X_test, y_train, y_test] = prepareData(features, labels, val_split)
    % Split data
    cv = cvpartition(length(labels), 'HoldOut', val_split);
    
    % Standardize features properly
    X_train = features(cv.training, :);
    y_train = labels(cv.training);
    X_test = features(cv.test, :);
    y_test = labels(cv.test);
    
    global mu sigma
    [X_train, mu, sigma] = zscore(X_train);
    X_test = (X_test - mu) ./ sigma;
end

function cnn_model = trainCNN(img_data, labels, img_size, numEpochs, doAugmentation)
    % Prepare image data
    X = cat(4, img_data{:});
    y = categorical(labels);
    
    % Split data
    cv = cvpartition(length(labels), 'HoldOut', 0.2);
    X_train = X(:,:,:,cv.training);
    y_train = y(cv.training);
    X_test = X(:,:,:,cv.test);
    y_test = y(cv.test);
    
    % Define architecture
    layers = [
        imageInputLayer([img_size(1) img_size(2) 3])
        convolution2dLayer(3, 16, 'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride',2)
        convolution2dLayer(3, 32, 'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride',2)
        fullyConnectedLayer(length(unique(labels)))
        softmaxLayer
        classificationLayer];
    
    % Training options
    options = trainingOptions('adam', ...
        'MaxEpochs', numEpochs, ...
        'ValidationData',{X_test, y_test}, ...
        'Plots','training-progress', ...
        'Verbose',true);
    
    % Data augmentation
    if doAugmentation
        augmenter = imageDataAugmenter(...
            'RandRotation',[-15 15], ...
            'RandXReflection',true);
        augimds = augmentedImageDatastore(img_size, X_train, y_train, ...
            'DataAugmentation',augmenter);
        cnn_model = trainNetwork(augimds, layers, options);
    else
        cnn_model = trainNetwork(X_train, y_train, layers, options);
    end
end

function knn_model = trainKNN(img_data, labels, val_split)
    % Extract features
    features = [];
    for i = 1:length(img_data)
        features = [features; extractCurrencyFeatures(img_data{i})];
    end
    
    % Split data
    cv = cvpartition(length(labels), 'HoldOut', val_split);
    
    % Train model
    knn_model = fitcknn(features(cv.training,:), labels(cv.training), ...
        'NumNeighbors',5, ...
        'Distance','cosine', ...
        'Standardize',true);
end

function saveModels(svm_model, cnn_model, knn_model, mu, sigma)
    save('currency_models.mat', 'svm_model', 'mu', 'sigma', '-v7.3');
    
    if ~isempty(cnn_model)
        save('denomination_cnn.mat', 'cnn_model', '-v7.3');
    elseif ~isempty(knn_model)
        save('denomination_knn.mat', 'knn_model', '-v7.3');
    end
end

function features = extractCurrencyFeatures(img)
    % Convert to grayscale
    grayImg = rgb2gray(img);
    
    % Basic texture features
    glcm = graycomatrix(grayImg);
    stats = graycoprops(glcm);
    
    % Color features
    hsv = rgb2hsv(img);
    colorFeatures = [mean2(hsv(:,:,1)), mean2(hsv(:,:,2)), mean2(hsv(:,:,3))];
    
    % Edge features
    edgeImg = edge(grayImg, 'canny');
    edgeDensity = sum(edgeImg(:))/numel(edgeImg);
    
    % Combine all features
    features = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity, ...
                mean2(grayImg), std2(grayImg), entropy(grayImg), ...
                colorFeatures, edgeDensity];
end

function testCurrencyImage(filename)
    global mu sigma final_svm_model cnn_model knn_model

    if isempty(final_svm_model) || isempty(mu) || isempty(sigma)
        errordlg('Models not loaded properly','Initialization Error');
        return;
    end

    try
        img = imresize(imread(filename), [256, 256]);
        features = extractCurrencyFeatures(img);
        features_std = (features - mu) ./ sigma;

        [pred_class, scores] = predict(final_svm_model, features_std);
        confidence = max(scores)*100;

        % OCR
        ocrResults = ocr(img, 'CharacterSet', '0123456789');
        ocrText = regexprep(ocrResults.Text, '\s+', '');
        detected_value = regexp(ocrText, '\d{3,4}', 'match');

        denom = 'Unknown';
        denom_conf = 0;
        denom_method = 'OCR';

        if ~isempty(detected_value)
            denom = detected_value{1};
            denom_conf = 95;
        else
            denom_method = 'CNN/KNN';
            if ~isempty(cnn_model)
                denom = classify(cnn_model, img);
                denom_scores = predict(cnn_model, img);
                denom_conf = max(denom_scores)*100;
            elseif ~isempty(knn_model)
                denom = predict(knn_model, features);
                [~, denom_scores] = predict(knn_model, features);
                denom_conf = max(denom_scores)*100;
            end
        end

        % Display results
        showResults(img, pred_class, confidence, denom, denom_conf, ocrResults, denom_method);

    catch ME
        errordlg(sprintf('Prediction failed:\n%s', ME.message),'Prediction Error');
    end
end


function showResults(img, pred_class, conf, denom, denom_conf, ocrResults, method)
    fig = figure('Name','Results','NumberTitle','off');

    % Original image with OCR overlay
    subplot(2,2,1);
    imshow(img);
    title('Input Image with OCR');
    
    % Annotate OCR results
    if ~isempty(ocrResults.Words)
        for k = 1:length(ocrResults.Words)
            if ~isempty(ocrResults.Words{k})
                rectangle('Position', ocrResults.WordBoundingBoxes(k,:), ...
                          'EdgeColor', 'g', 'LineWidth', 1.5);
                text(double(ocrResults.WordBoundingBoxes(k,1)), ...
                     double(ocrResults.WordBoundingBoxes(k,2)) - 5, ...
                     ocrResults.Words{k}, 'Color','green','FontSize',10, ...
                     'FontWeight','bold');
            end
        end
    end

    % Segmentation
    [BW, maskedImg] = createMask(img);
    subplot(2,2,2);
    imshow(BW);
    title('Segmentation');

    subplot(2,2,3);
    imshow(maskedImg);
    title('Processed');

    % Results display
    auth = {'Genuine','Fake'};
    resultStr = sprintf('%s (%.1f%%)\nDenomination: %s (%.1f%%)\nMethod: %s', ...
                       auth{pred_class}, conf, denom, denom_conf, method);

    subplot(2,2,4);
    text(0.1, 0.5, resultStr, 'FontSize',12, 'FontWeight','bold');
    axis off;
end


function [BW, maskedImg] = createMask(RGB)
    hsv = rgb2hsv(RGB);
    
    % Adaptive thresholding
    BW = (hsv(:,:,2) > 0.1) & (hsv(:,:,3) > 0.2);
    
    % Clean up mask
    BW = imfill(BW,'holes');
    BW = bwareaopen(BW, 100);
    
    maskedImg = RGB;
    maskedImg(repmat(~BW,[1 1 3])) = 0;
end

function runCurrencyDetection()
    fig = figure('Name','Currency Detector','Position',[300 300 400 300]);
    
    uicontrol('Style','pushbutton', 'String','Select Image', ...
              'Position',[150 200 100 30], 'Callback',@selectImage);
    
    uicontrol('Style','text', 'String','Select currency image to analyze', ...
              'Position',[50 150 300 30], 'Tag','statusText');
    
    function selectImage(~,~)
        [file, path] = uigetfile({'*.jpg;*.png','Image Files'});
        if file ~= 0
            set(findobj('Tag','statusText'), 'String','Processing...');
            drawnow;
            testCurrencyImage(fullfile(path, file));
            set(findobj('Tag','statusText'), 'String','Done. Select another image?');
        end
    end
end