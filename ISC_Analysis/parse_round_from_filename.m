function r = parse_round_from_filename(filename)

r = nan;

% attempt_06
tok = regexp(filename, 'attempt[_-]?(\d+)', 'tokens', 'once', 'ignorecase');
if ~isempty(tok)
    r = str2double(tok{1});
    return;
end

% round_06
tok = regexp(filename, 'round[_-]?(\d+)', 'tokens', 'once', 'ignorecase');
if ~isempty(tok)
    r = str2double(tok{1});
    return;
end

% r6
tok = regexp(filename, '(?:^|[_-])r(\d+)', 'tokens', 'once', 'ignorecase');
if ~isempty(tok)
    r = str2double(tok{1});
    return;
end

error("Could not extract round number from filename: %s", filename);
end
