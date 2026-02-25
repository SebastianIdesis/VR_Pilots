function s = parse_subject_from_filename(filename)
%PARSE_SUBJECT_FROM_FILENAME Extract subject ID from filename.
% Matches: subject12 / subj12 / sub12 / s12 (with optional _ or -)

s = nan;

tok = regexp(filename, '(subject|subj|sub|s)[_-]?(\d+)', ...
             'tokens', 'once', 'ignorecase');
if ~isempty(tok)
    s = str2double(tok{2});
end

end
