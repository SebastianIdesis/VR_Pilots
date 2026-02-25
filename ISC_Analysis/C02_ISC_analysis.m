%% First Run the Prepare data to have the signals ready

%% Step 3: Compute ISC on all samples
fprintf("Computing ISC...\n");
[ISC, ISC_persubject, ISC_persecond, W, A] = ISC_illuminate(EEG_3D(:,:,:));


% Add ISC values to sample metadata
sampleMeta.ISC_Component1 = ISC_persubject(1, :)';
sampleMeta.ISC_Component2 = ISC_persubject(2, :)';
sampleMeta.ISC_Component3 = ISC_persubject(3, :)';
sampleMeta.ISC_Component4 = ISC_persubject(4, :)';
sampleMeta.ISC_Component5 = ISC_persubject(5, :)';
sampleMeta.ISC_Component6 = ISC_persubject(6, :)';
sampleMeta.ISC_Component6 = ISC_persubject(7, :)';
sampleMeta.ISC_Component6 = ISC_persubject(8, :)';
sampleMeta.ISC_Component6 = ISC_persubject(9, :)';
sampleMeta.ISC_Component6 = ISC_persubject(10, :)';

%% Step 4: Compare ISC values by quality
qualities = unique(sampleMeta.Quality);
data_matrix = [];

for i = 1:length(qualities)
    q = qualities{i};
    values = sampleMeta.ISC_Component1(strcmp(sampleMeta.Quality, q));
    data_matrix = [data_matrix values]; %#ok<AGROW>
end

% Friedman test
[p, tbl, stats] = friedman(data_matrix, 1);
fprintf('Friedman test p-value: %.4f\n', p);

% Boxplot
figure;
boxplot(data_matrix, 'Labels', qualities);
title('ISC by Quality Level');
ylabel('ISC (Component 1)');
saveas(gcf, 'ISC_Processed_Quality_Comparison_Boxplot_15_sec.png');

% Save table
writetable(sampleMeta, 'ISC_Processed_Results_PerSample_15sec.csv');