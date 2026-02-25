% Build EEG_3D: [time_samples x channels x nRecordings]
% Assumptions:
%   - Each .mat contains a struct field: EEG
%   - EEG.data holds [time_samples x channels]
%   - (Optional) EEG.t exists, but we only stack EEG.data
% Notes:
%   - If segments have different lengths, this code pads with NaN to the
%     maximum length (so you still get a clean 3D matrix).

% --------- USER: set your folder ---------
inDir = "Data";  % <-- change
% ----------------------------------------

files = dir(fullfile(inDir, "*.mat"));
if isempty(files)
    error("No .mat files found in: %s", inDir);
end

% Natural sort by filename if available (optional)
% If you don't have natsortfiles, just comment this out.
try
    files = natsortfiles(files);
catch
    % no-op
end

nRec = numel(files);

% First pass: read sizes, validate, find max length/channels
lens = zeros(nRec,1);
nChs = zeros(nRec,1);

for i = 1:nRec
    S = load(fullfile(files(i).folder, files(i).name));
    if ~isfield(S, "EEG") || ~isfield(S.EEG, "data")
        error("File %s does not contain EEG.data", files(i).name);
    end

    X = S.EEG.data;
    if ~isnumeric(X) || ndims(X) ~= 2
        error("EEG.data in %s must be a numeric 2D matrix [time x channels].", files(i).name);
    end

    lens(i) = size(X,1);
    nChs(i) = size(X,2);
end

% Check channel consistency
if numel(unique(nChs)) ~= 1
    error("Channel count differs across files: %s", mat2str(unique(nChs)'));
end
nChannels = nChs(1);

% Choose stacking length (pad to max; or change to min to truncate)
T = max(lens);      % pad to longest
% T = min(lens);    % uncomment if you prefer truncation to shortest

% Preallocate output (NaN padding)
eegMatrix = nan(T, nChannels, nRec);

% Optional: store filenames + attempts/metadata if present
fileNames = strings(nRec,1);
attempts  = nan(nRec,1);

% Second pass: fill 3D array
for i = 1:nRec
    fileNames(i) = string(files(i).name);
    S = load(fullfile(files(i).folder, files(i).name));
    X = S.EEG.data;

    % If truncating to min length:
    X = X(1:min(size(X,1),T), :);

    eegMatrix(1:size(X,1), :, i) = X;

    % Optional: attempt field if you saved it at top-level
    if isfield(S, "attempt")
        attempts(i) = double(S.attempt);
    end
end

EEG_3D = eegMatrix;
% EEG_3D is your final variable: [time_samples x channels x recordings]
disp(size(eegMatrix));

% Optional: package into one struct for saving
EEG_stack = struct();
EEG_stack.data = eegMatrix;
EEG_stack.fileNames = fileNames;
EEG_stack.attempts = attempts;

% save(fullfile(inDir, "EEG_stacked.mat"), "EEG_stack", "-v7.3");
