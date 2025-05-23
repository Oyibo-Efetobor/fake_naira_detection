% filepath: c:\Users\efetobordev\Desktop\trial\gui.m
function gui()
    % Nigerian Currency Detection GUI (uses pre-trained models)
    % Loads models and provides an interface for testing images

    % Load models
    models = loadModels();

    % Create GUI
    fig = figure('Name','Naira Note Scanner','Position',[300 300 400 300]);
    uicontrol('Style','pushbutton', 'String','Select Image', ...
        'Position',[150 200 100 30], 'Callback',@selectImage);

    uicontrol('Style','text', 'String','Select currency image to analyze', ...
        'Position',[50 150 300 30], 'Tag','statusText');

    function selectImage(~,~)
        [file, path] = uigetfile({'*.jpg;*.png','Image Files'});
        if file ~= 0
            set(findobj('Tag','statusText'), 'String','Processing...');
            drawnow;
            testCurrencyImage_gui(fullfile(path, file), models);
            set(findobj('Tag','statusText'), 'String','Done. Select another image?');
        end
    end
end

function models = loadModels()
    % Load SVM and normalization params
    s = load('currency_models.mat', 'svm_model', 'mu', 'sigma');
    models.svm_model = s.svm_model;
    models.mu = s.mu;
    models.sigma = s.sigma;

    % Try to load CNN or KNN for denomination
    if exist('denomination_cnn.mat','file')
        c = load('denomination_cnn.mat', 'cnn_model');
        models.cnn_model = c.cnn_model;
        models.knn_model = [];
    elseif exist('denomination_knn.mat','file')
        k = load('denomination_knn.mat', 'knn_model');
        models.cnn_model = [];
        models.knn_model = k.knn_model;
    else
        models.cnn_model = [];
        models.knn_model = [];
    end
end

function testCurrencyImage_gui(filename, models)
    % Use the same logic as in trial.m, but with loaded models
    try
        img = imresize(imread(filename), [256, 256]);
        features = extractCurrencyFeatures(img);
        features_std = (features - models.mu) ./ models.sigma;

        [pred_class, scores] = predict(models.svm_model, features_std);
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
            if ~isempty(models.cnn_model)
                denom = classify(models.cnn_model, img);
                denom_scores = predict(models.cnn_model, img);
                denom_conf = max(denom_scores)*100;
            elseif ~isempty(models.knn_model)
                denom = predict(models.knn_model, features);
                [~, denom_scores] = predict(models.knn_model, features);
                denom_conf = max(denom_scores)*100;
            end
        end

        % Display results
        showResults(img, pred_class, confidence, denom, denom_conf, ocrResults, denom_method);

    catch ME
        errordlg(sprintf('Prediction failed:\n%s', ME.message),'Prediction Error');
    end
end

function features = extractCurrencyFeatures(img)
    % Same as in trial.m
    grayImg = rgb2gray(img);
    glcm = graycomatrix(grayImg);
    stats = graycoprops(glcm);
    hsv = rgb2hsv(img);
    colorFeatures = [mean2(hsv(:,:,1)), mean2(hsv(:,:,2)), mean2(hsv(:,:,3))];
    edgeImg = edge(grayImg, 'canny');
    edgeDensity = sum(edgeImg(:))/numel(edgeImg);
    features = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity, ...
                mean2(grayImg), std2(grayImg), entropy(grayImg), ...
                colorFeatures, edgeDensity];
end

function showResults(img, pred_class, conf, denom, denom_conf, ocrResults, method)
    % Create a very large, centered figure for maximum visibility
    screenSize = get(0, 'ScreenSize');
    figWidth = min(1800, screenSize(3) - 40); % nearly full width
    figHeight = min(1000, screenSize(4) - 40); % nearly full height
    figPos = [ (screenSize(3)-figWidth)/2, (screenSize(4)-figHeight)/2, figWidth, figHeight ];
    fig = figure('Name','Results','NumberTitle','off','Position',figPos);
    
    subplot(2,2,1);
    imshow(img);
    title('Input Image with OCR','FontSize',22);
    if ~isempty(ocrResults.Words)
        for k = 1:length(ocrResults.Words)
            if ~isempty(ocrResults.Words{k})
                rectangle('Position', ocrResults.WordBoundingBoxes(k,:), ...
                          'EdgeColor', 'g', 'LineWidth', 3);
                text(double(ocrResults.WordBoundingBoxes(k,1)), ...
                     double(ocrResults.WordBoundingBoxes(k,2)) - 5, ...
                     ocrResults.Words{k}, 'Color','green','FontSize',18, ...
                     'FontWeight','bold');
            end
        end
    end
    [BW, maskedImg] = createMask(img);
    subplot(2,2,2);
    imshow(BW);
    title('Segmentation','FontSize',22);
    subplot(2,2,3);
    imshow(maskedImg);
    title('Processed','FontSize',22);
    auth = {'Genuine','Fake'};
    resultStr = sprintf('%s (%.1f%%)\nDenomination: %s (%.1f%%)\nMethod: %s', ...
                       auth{pred_class}, conf, denom, denom_conf, method);
    subplot(2,2,4);
    text(0.1, 0.5, resultStr, 'FontSize',26, 'FontWeight','bold');
    axis off;
end

function [BW, maskedImg] = createMask(RGB)
    hsv = rgb2hsv(RGB);
    BW = (hsv(:,:,2) > 0.1) & (hsv(:,:,3) > 0.2);
    BW = imfill(BW,'holes');
    BW = bwareaopen(BW, 100);
    maskedImg = RGB;
    maskedImg(repmat(~BW,[1 1 3])) = 0;
end