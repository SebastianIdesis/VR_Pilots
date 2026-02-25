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

EEG_3D = EEG_3D(:,1:32,:);
%%
%% =========================
% SPLIT EEG_3D IN HALF (TIME)
%% =========================
% EEG_3D is assumed: [T x Ch x N]

T = size(EEG_3D, 1);
midT = floor(T/2);

EEG_3D1 = EEG_3D(1:midT, :, :);        % first half
EEG_3D2 = EEG_3D(midT+1:end, :, :);    % second half

%% ==========================================
% HELPER: Compute ISC per round for an EEG_3D
%% ==========================================
function ISC_byRound = compute_ISC_perRound(EEG_3D_in, roundIdx, subjId, FS, doTrimToCommonLength)
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

        Xr = EEG_3D_in(:, :, idx);

        if doTrimToCommonLength
            % validT: timepoints where ALL ch and ALL rec are non-NaN
            validT = squeeze(all(all(~isnan(Xr), 2), 3));
            lastValid = find(validT, 1, 'last');

            if isempty(lastValid) || lastValid < 2
                fprintf("Round %d skipped (no valid common timepoints)\n", r);
                continue;
            end

            Xr = Xr(1:lastValid, :, :);
        end

        [ISC, ISC_persubject, ISC_persecond, W, A] = isceeg(Xr, FS);

        outRow = outRow + 1;
        ISC_byRound(outRow).round = r;
        ISC_byRound(outRow).nRecordings = numel(idx);
        ISC_byRound(outRow).ISC = ISC;
        ISC_byRound(outRow).ISC_persubject = ISC_persubject;
        ISC_byRound(outRow).ISC_persecond = ISC_persecond;
        ISC_byRound(outRow).subjectIds = subjId(idx);
        ISC_byRound(outRow).W = W;
        ISC_byRound(outRow).A = A;

        fprintf("Round %d → ISC = %g\n", r, ISC(1));
    end
end

%% ======================================
% RUN ISC PER ROUND FOR EACH HALF
%% ======================================
ISC_byRound_H1 = compute_ISC_perRound(EEG_3D1, roundIdx, subjId, FS, doTrimToCommonLength);
ISC_byRound_H2 = compute_ISC_perRound(EEG_3D2, roundIdx, subjId, FS, doTrimToCommonLength);

%% ======================
% PLOTTING FUNCTION
%% ======================
function plot_ISC_results(ISC_byRound, plotTitlePrefix)
    if isempty(ISC_byRound)
        warning("%s: ISC_byRound is empty. Nothing to plot.", plotTitlePrefix);
        return;
    end

    roundList = [ISC_byRound.round];
    ISC1 = arrayfun(@(x) x.ISC(1), ISC_byRound);

    % Line plot
    figure;
    plot(roundList, ISC1, "-o");
    xlabel("Round"); ylabel("ISC (Component 1)");
    title(plotTitlePrefix + " — ISC per Round");
    grid on;

    % Dot plot
    figure;
    scatter(roundList, ISC1, 60, "filled"); hold on
    plot(roundList, ISC1, "-");
    xlabel("Round"); ylabel("ISC (Component 1)");
    title(plotTitlePrefix + " — ISC (1st component) per Round");
    grid on;

    % "Violin-ish" (box + swarm) plot using per-subject ISC
    y = [];
    g = [];
    for i = 1:numel(ISC_byRound)
        perSub = ISC_byRound(i).ISC_persubject;
        y = [y; perSub(:)];
        g = [g; repmat(roundList(i), numel(perSub), 1)];
    end

    figure;
    boxchart(g, y); hold on
    swarmchart(g, y, 18, "filled");
    xlabel("Round"); ylabel("ISC per subject (Component 1)");
    title(plotTitlePrefix + " — ISC per subject by Round (Box + Dots)");
    grid on;
end

%% ======================
% MAKE PLOTS FOR EACH HALF
%% ======================
plot_ISC_results(ISC_byRound_H1, "First half");
plot_ISC_results(ISC_byRound_H2, "Second half");

%% In one plot :
round1 = [ISC_byRound_H1.round];
isc1   = arrayfun(@(x) x.ISC(1), ISC_byRound_H1);

round2 = [ISC_byRound_H2.round];
isc2   = arrayfun(@(x) x.ISC(1), ISC_byRound_H2);

figure;
plot(round1, isc1, "-o", "LineWidth", 1.5); hold on
plot(round2, isc2, "-s", "LineWidth", 1.5);
xlabel("Round"); ylabel("ISC (Component 1)");
title("ISC per Round: First half vs Second half");
legend("First half","Second half","Location","best");
ylim([0 0.2])
grid on

%%
figure;
scatter(round1, isc1, 60, "filled"); hold on
scatter(round2, isc2, 60, "filled");
xlabel("Round"); ylabel("ISC (Component 1)");
title("ISC per Round (dots): First half vs Second half");
legend("First half","Second half","Location","best");
grid on