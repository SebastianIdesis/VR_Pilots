inDir = "Data"; % <-- change
doTrimToCommonLength = true;
usePaddingToMaxLength = true;  % true: pad to max with NaN, false: truncate to min


files = dir(fullfile(inDir, "*.mat"));

nFiles = numel(files);

% ---------------- PASS 1: sizes + IDs ----------------
lens = zeros(nFiles,1);
nChs = zeros(nFiles,1);
subjId = nan(nFiles,1);
roundIdx = nan(nFiles,1);
fileNames = strings(nFiles,1);

for i = 1:nFiles
    fileNames(i) = string(files(i).name);
    fp = fullfile(files(i).folder, files(i).name);
    S = load(fp);

    % Validate structure
    if ~isfield(S,"EEG") || ~isfield(S.EEG,"data")
        error("File %s missing EEG.data", files(i).name);
    end

    X = S.EEG.data;
    if ~isnumeric(X) || ndims(X) ~= 2
        error("EEG.data in %s must be numeric 2D [time x channels].", files(i).name);
    end

    lens(i) = size(X,1);
    nChs(i) = size(X,2);

    % Parse IDs from filename: p{subject}_round_{round}.mat
    [subjId(i), roundIdx(i)] = parse_p_round_filename(files(i).name);
end

if any(isnan(subjId))
    bad = find(isnan(subjId));
    error("Could not parse subject id for: %s", strjoin(fileNames(bad), ", "));
end
if any(isnan(roundIdx))
    bad = find(isnan(roundIdx));
    error("Could not parse round id for: %s", strjoin(fileNames(bad), ", "));
end

if numel(unique(nChs)) ~= 1
    error("Channel count differs across files: %s", mat2str(unique(nChs)'));
end
nChannels = nChs(1);

% Decide T
if usePaddingToMaxLength
    T = max(lens);
else
    T = min(lens);
end

% ---------------- PASS 2: Build EEG_3D ----------------
EEG_3D = nan(T, nChannels, nFiles);

for i = 1:nFiles
    S = load(fullfile(files(i).folder, files(i).name));
    X = S.EEG.data;

    if usePaddingToMaxLength
        X = X(1:min(size(X,1),T), :);
        EEG_3D(1:size(X,1), :, i) = X;
    else
        EEG_3D(:, :, i) = X(1:T, :);
    end
end

% Optional: trim to last timepoint where ALL values exist (removes NaN padding)
if doTrimToCommonLength
    validT = squeeze(all(all(~isnan(EEG_3D),2),3));  % [T x 1]
    lastValid = find(validT, 1, "last");
    EEG_3D = EEG_3D(1:lastValid,:,:);
end

% Show mapping
disp(table(fileNames, subjId, roundIdx, lens, ...
    'VariableNames', ["file","subject","round","nSamples"]));

% Check balanced design (3 subjects x 6 rounds)
subjects = sort(unique(subjId(:)));
rounds = sort(unique(roundIdx(:)));
fprintf("Subjects: %s\n", mat2str(subjects'));
fprintf("Rounds:   %s\n", mat2str(rounds'));

for s = subjects'
    for r = rounds'
        n_sr = sum(subjId==s & roundIdx==r);
        if n_sr ~= 1
            error("Expected exactly 1 trial for Subject %d Round %d, found %d.", s, r, n_sr);
        end
    end
end

% ---------------- Compute ISC once across ALL trials ----------------
[ISC_all, ISC_persubject_all, ISC_persecond_all, W, A] = ISC_illuminate(EEG_3D);

% Extract one ISC value per trial (component 1 if multiple)
if isvector(ISC_persubject_all)
    ISC_trial = ISC_persubject_all(:);
else
    ISC_trial = ISC_persubject_all(:,1);
end

% ---------------- Compare ISC by round (within-subject) ----------------
Tbl = table;
Tbl.ISC = double(ISC_trial);
Tbl.Subject = categorical(subjId);
Tbl.Round = categorical(roundIdx);

% Subject x Round matrix
ISC_SR = nan(numel(subjects), numel(rounds));
for si = 1:numel(subjects)
    for ri = 1:numel(rounds)
        ISC_SR(si,ri) = Tbl.ISC(subjId==subjects(si) & roundIdx==rounds(ri));
    end
end

% Plot trajectories
figure;
plot(rounds, ISC_SR', '-o'); hold on
plot(rounds, mean(ISC_SR,1,'omitnan'), 'k-', 'LineWidth', 2);
xlabel("Round"); ylabel("ISC per trial (comp1 if multi-comp)");
title("ISC by round (each line=subject, black=mean)");
grid on; hold off

% Mixed-effects model (recommended)
lme = fitlme(Tbl, 'ISC ~ Round + (1|Subject)');
disp("Mixed effects ANOVA (Round effect):");
disp(anova(lme));

% Repeated-measures ANOVA (classic)
varNames = "R" + string(rounds); % R1..R6
wide = array2table(ISC_SR, 'VariableNames', varNames);
wide.Subject = categorical(subjects(:));
within = table(categorical(rounds(:)), 'VariableNames', "Round");

rm = fitrm(wide, sprintf('%s-%s ~ 1', varNames(1), varNames(end)), 'WithinDesign', within);
disp("Repeated measures ANOVA (Round effect):");
disp(ranova(rm, 'WithinModel', 'Round'));

%% ---------- local function ----------
function [subj, rnd] = parse_p_round_filename(fn)
% Parse: p{subject}_round_{round}.mat  (case-insensitive)
% Examples: p1_round_1.mat, p03_round_06.mat

tok = regexp(fn, '^p(\d+)_round_(\d+)\.mat$', 'tokens', 'once', 'ignorecase');
if isempty(tok)
    subj = nan; rnd = nan;
else
    subj = str2double(tok{1});
    rnd  = str2double(tok{2});
end
end
