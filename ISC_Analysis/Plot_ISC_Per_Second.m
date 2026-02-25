%% ---------------- Plot ISC_persecond to compare rounds ----------------
% Compute ISC_persecond separately for each round (across subjects)
rounds = sort(unique(roundIdx(:)));
nR = numel(rounds);
nR = 3;

ISCps_round = cell(nR,1);     % each cell: [nComp x nSec] (or similar)
nSec_round  = zeros(nR,1);

for ri = 1:nR
    r = rounds(ri);
    idx = find(roundIdx == r);             % trials belonging to this round (should be 3)
    EEG_r = EEG_3D(:,:,idx);

    % Optional safety: drop any remaining NaNs (if you didn't trim)
    validT = squeeze(all(all(~isnan(EEG_r),2),3));
    lastValid = find(validT, 1, "last");
    EEG_r = EEG_r(1:lastValid,:,:);

    [ISC_r, ISC_psub_r, ISC_ps_r] = isceeg(EEG_r, FS);

    ISCps_round{ri} = ISC_ps_r;
    nSec_round(ri)  = size(ISC_ps_r, 2);   % assumes [nComp x nSec]
end

% Make all rounds same number of seconds (truncate to shortest)
minSec = min(nSec_round);
for ri = 1:nR
    ISCps_round{ri} = ISCps_round{ri}(:, 1:minSec);
end

% Choose which component to plot (usually comp 1 is most used)
compToPlot = 1;

% Stack into [nR x minSec] for plotting
ISCps_mat = nan(nR, minSec);
for ri = 1:nR
    ISCps_mat(ri,:) = ISCps_round{ri}(compToPlot, :);
end

tsec = 1:minSec;

% 1) Overlay curves (one per round)
figure;
plot(tsec, ISCps_mat', 'LineWidth', 1); hold on
plot(tsec, mean(ISCps_mat,1,'omitnan'), 'k-', 'LineWidth', 2);
xlabel("Time (s)"); ylabel(sprintf("ISC_persecond (comp %d)", compToPlot));
title("ISC per second by round (each line = round, black = mean across rounds)");
grid on; hold off

% 2) Heatmap (round x time), great for comparison
figure;
imagesc(tsec, rounds, ISCps_mat); axis xy
xlabel("Time (s)"); ylabel("Round");
title(sprintf("ISC_persecond heatmap (comp %d)", compToPlot));
colorbar