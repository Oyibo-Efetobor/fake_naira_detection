% filepath: c:\Users\efetobordev\Desktop\trial\store_test.m
% Automatically scan dataset images and store correctly detected denomination images

% Load models
models = loadModels();

dataset_dir = 'datasets';
output_dir = 'test_images';

classes = {'genuine', 'fake'};

for ci = 1:length(classes)
    class_dir = fullfile(dataset_dir, classes{ci});
    denom_dirs = dir(class_dir);
    denom_dirs = denom_dirs([denom_dirs.isdir] & ~ismember({denom_dirs.name}, {'.','..'}));
    for di = 1:length(denom_dirs)
        denom_name = denom_dirs(di).name;
        denom_val = str2double(denom_name);
        img_files = dir(fullfile(class_dir, denom_name, '*.jpg'));
        for fi = 1:length(img_files)
            img_path = fullfile(class_dir, denom_name, img_files(fi).name);
            try
                img = imresize(imread(img_path), [256, 256]);
                % Use denomination detection logic from gui.m
                detected_denom = detectDenomination(img, models);
                if isequal(str2double(detected_denom), denom_val)
                    % Prepare output folder
                    out_folder = fullfile(output_dir, classes{ci}, denom_name);
                    if ~exist(out_folder, 'dir')
                        mkdir(out_folder);
                    end
                    % Copy image
                    copyfile(img_path, fullfile(out_folder, img_files(fi).name));
                end
            catch
                % Ignore errors and continue
            end
        end
    end
end

disp('Done storing correctly detected images.');

function models = loadModels()
    s = load('currency_models.mat', 'svm_model', 'mu', 'sigma');
    models.svm_model = s.svm_model;
    models.mu = s.mu;
    models.sigma = s.sigma;
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

function denom = detectDenomination(img, models)
    % OCR first
    ocrResults = ocr(img, 'CharacterSet', '0123456789');
    ocrText = regexprep(ocrResults.Text, '\s+', '');
    detected_value = regexp(ocrText, '\d{3,4}', 'match');
    if ~isempty(detected_value)
        denom = detected_value{1};
        return;
    end
    % CNN/KNN fallback
    if ~isempty(models.cnn_model)
        denom = char(classify(models.cnn_model, img));
    elseif ~isempty(models.knn_model)
        features = extractCurrencyFeatures(img);
        denom = num2str(predict(models.knn_model, features));
    else
        denom = 'Unknown';
    end
end

function features = extractCurrencyFeatures(img)
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