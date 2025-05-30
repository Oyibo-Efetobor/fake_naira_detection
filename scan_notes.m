% filepath: c:\Users\efetobordev\Desktop\trial\gui.m
function gui()
    % Nigerian Currency Detection GUI (uses pre-trained models)
    % Loads models and provides an interface for testing images

    % Load models
    models = loadModels();

    % Get screen size for full screen
    screenSize = get(0, 'ScreenSize');
    figWidth = screenSize(3);
    figHeight = screenSize(4);
    figPos = [1 1 figWidth figHeight];

    % Create GUI with enhanced UI, now full screen
    fig = figure('Name','FAKE NAIRA CURRENCY DETECTION APP', ...
        'Position',figPos, ...
        'Color',[0.13 0.18 0.24], ...
        'NumberTitle','off', ...
        'MenuBar','none', ...
        'WindowState','maximized');

    % Centered title and subtitle
    titleHeight = figHeight*0.06;
    titleWidth = figWidth*0.7;
    titleX = (figWidth-titleWidth)/2;
    titleY = figHeight/2 + figHeight*0.18;
    uicontrol('Style','text', 'String','FAKE NAIRA CURRENCY DETECTION APP', ...
        'FontSize',round(figHeight*0.032), 'FontWeight','bold', ...
        'ForegroundColor',[0.95 0.95 0.95], ...
        'BackgroundColor',[0.13 0.18 0.24], ...
        'HorizontalAlignment','center', ...
        'Position',[titleX titleY titleWidth titleHeight]);

    subtitleHeight = figHeight*0.04;
    subtitleWidth = figWidth*0.5;
    subtitleX = (figWidth-subtitleWidth)/2;
    subtitleY = titleY - subtitleHeight - 10;
    uicontrol('Style','text', 'String','Scan and verify your Naira notes', ...
        'FontSize',round(figHeight*0.022), 'FontWeight','bold', ...
        'ForegroundColor',[0.8 0.8 0.8], ...
        'BackgroundColor',[0.13 0.18 0.24], ...
        'HorizontalAlignment','center', ...
        'Position',[subtitleX subtitleY subtitleWidth subtitleHeight]);

    % Center the main controls vertically
    centerY = figHeight/2;
    buttonHeight = figHeight*0.08;
    buttonWidth = figWidth*0.18;
    buttonY = centerY - buttonHeight/2;
    uicontrol('Style','pushbutton', 'String','Select Image', ...
        'FontSize',round(figHeight*0.03), 'FontWeight','bold', ...
        'BackgroundColor',[0.2 0.6 0.3], ...
        'ForegroundColor',[1 1 1], ...
        'Position',[figWidth/2-buttonWidth/2 buttonY buttonWidth buttonHeight], ...
        'Callback',@selectImage, ...
        'CData',[]);

    % Status Text (just below the button, centered)
    statusHeight = figHeight*0.045;
    statusWidth = figWidth*0.36;
    statusX = (figWidth-statusWidth)/2;
    statusY = buttonY - statusHeight - 20;
    uicontrol('Style','text', 'String','Select a currency image to analyze', ...
        'FontSize',round(figHeight*0.018), ...
        'ForegroundColor',[0.9 0.9 0.9], ...
        'BackgroundColor',[0.13 0.18 0.24], ...
        'HorizontalAlignment','center', ...
        'Position',[statusX statusY statusWidth statusHeight], 'Tag','statusText');

    % Footer (lower position)
    footerHeight = figHeight*0.03;
    uicontrol('Style','text', 'String','Powered by MATLAB & Machine Learning', ...
        'FontSize',round(figHeight*0.015), ...
        'ForegroundColor',[0.7 0.7 0.7], ...
        'BackgroundColor',[0.13 0.18 0.24], ...
        'Position',[figWidth/2-figWidth*0.15 10 figWidth*0.3 footerHeight]);

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