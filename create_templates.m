% create_templates.m
% Script to create denomination templates for Nigerian Naira detection system

function create_templates()
    % Check if templates folder exists
    if ~exist('templates', 'dir')
        mkdir('templates');
        disp('Created templates directory');
    end

    % List of required denominations
    denominations = {'200', '500', '1000'};
    template_struct = struct();
    
    disp('Creating denomination templates...');
    
    for i = 1:length(denominations)
        denom = denominations{i};
        filename = fullfile('templates', [denom '.jpg']);
        
        if exist(filename, 'file')
            % Load and process template image
            try
                img = imread(filename);
                
                % Convert to grayscale and binarize
                gray = rgb2gray(img);
                binary = imbinarize(gray);
                
                % Clean up the image
                binary = bwareaopen(binary, 50); % Remove small objects
                binary = imfill(binary, 'holes');
                
                % Store in structure
                field_name = ['denom_' denom];
                template_struct.(field_name) = binary;
                
                disp(['Successfully processed template for ' denom ' Naira']);
                
                % Show the template for verification
                figure;
                imshow(binary);
                title(['Template for ' denom ' Naira']);
                
            catch ME
                warning('Failed to process %s: %s', filename, ME.message);
            end
        else
            warning('Template image not found: %s', filename);
            disp('Please add a clean image of the denomination number to the templates folder');
        end
    end
    
    % Save templates to file
    if ~isempty(fieldnames(template_struct))
        save('denomination_templates.mat', '-struct', 'template_struct');
        disp('Successfully saved templates to denomination_templates.mat');
    else
        error('No valid templates were processed');
    end
end

% Execute the function
create_templates();