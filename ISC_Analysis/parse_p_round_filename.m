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
