clc;clear all;
inDir = "Data"; 
FS = 500;
doTrimToCommonLength = true;
usePaddingToMaxLength = true;


files = dir(fullfile(inDir, "*.mat"));

nFiles = numel(files);

lens = zeros(nFiles,1);
nChs = zeros(nFiles,1);
roundIdx = nan(nFiles,1);
subjId = nan(nFiles,1);
fileNames = strings(nFiles,1);

%% PASS 1: read metadata
for i = 1:nFiles
    fileNames(i) = string(files(i).name);
    S = load(fullfile(files(i).folder, files(i).name));

    X = S.EEG.data;
    lens(i) = size(X,1);
    nChs(i) = size(X,2);

    if isfield(S,"attempt") && ~isempty(S.attempt)
        roundIdx(i) = double(S.attempt);
    else
        roundIdx(i) = parse_round_from_filename(files(i).name);
    end

    subjId(i) = parse_subject_from_filename(files(i).name);
end

if any(isnan(roundIdx))
    error("Some files have missing round numbers.");
end

if numel(unique(nChs)) ~= 1
    error("Channel mismatch across files.");
end

nChannels = nChs(1);

if usePaddingToMaxLength
    T = max(lens);
else
    T = min(lens);
end

%% PASS 2: build EEG_3D
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

EEG_3D = EEG_3D(:,:,:);
%% COMPUTE ISC PER ROUND

rounds = unique(roundIdx);
rounds = sort(rounds);

ISC_byRound = struct();
outRow = 0;

for k = 1:numel(rounds)
    r = rounds(k);
    idx = find(roundIdx == r);

    if numel(idx) < 2
        fprintf("Round %d skipped (only %d recordings)\n", r, numel(idx));
        continue;
    end

    Xr = EEG_3D(:, :, idx);

    if doTrimToCommonLength
        validT = squeeze(all(all(~isnan(Xr),2),3));
        lastValid = find(validT,1,'last');
        Xr = Xr(1:lastValid,:,:);
    end

    [ISC, ISC_persubject, ISC_persecond, W, A] = isceeg(Xr, FS);
    %[ISC,ISC_persubject,ISC_persecond,W,A,p_ISC,p_ISC_persecond,null] = isceeg(Xr, FS);
    outRow = outRow + 1;
    ISC_byRound(outRow).round = r;
    ISC_byRound(outRow).nRecordings = numel(idx);
    ISC_byRound(outRow).ISC = ISC;
    ISC_byRound(outRow).ISC_persubject = ISC_persubject;
    ISC_byRound(outRow).ISC_persecond = ISC_persecond;
    ISC_byRound(outRow).subjectIds = subjId(idx);

    fprintf("Round %d → ISC = %g\n", r, ISC(1));
end

%% Plot (first component if multi-dimensional ISC)
figure;
roundList = [ISC_byRound.round];
ISCvals = arrayfun(@(x) x.ISC(1), ISC_byRound);
plot(roundList, ISCvals, "-o");
xlabel("Round"); ylabel("ISC (Component 1)");
title("ISC per Round");
grid on;
%% Dot Plot
roundList = [ISC_byRound.round];
ISC1     = arrayfun(@(x) x.ISC(1), ISC_byRound);

figure;
scatter(roundList, ISC1, 60, "filled"); hold on
plot(roundList, ISC1, "-"); % optional: connect the dots
xlabel("Round"); ylabel("ISC (Component 1)");
title("ISC (1st component) per Round");
grid on
ylim([0 0.15])

%% Violin Plot
roundList = [ISC_byRound.round];

y = [];
g = [];
for i = 1:numel(ISC_byRound)
    perSub = ISC_byRound(i).ISC_persubject;
    y = [y; perSub(:)];
    g = [g; repmat(roundList(i), numel(perSub), 1)];
end

figure;
boxchart(g, y); hold on
swarmchart(g, y, 18, "filled"); % dot cloud on top
xlabel("Round"); ylabel("ISC per subject (Component 1)");
title("ISC per subject by Round (Box + Dots)");
grid on