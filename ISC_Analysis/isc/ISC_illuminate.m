function [ISC,ISC_persubject,ISC_persecond,W,A] = ISC_illuminate(X, fs)
    % some ISC processing parameters
    gamma = 0.5; % shrinkage parameter; smaller gamma for less regularization
    Nsec  = 15;  % time-window (in seconds) over which to compute time-reposeved ISC
    Ncomp = 3;  % number of components to dispaly (all D are computed)

    %     if exist(datafile) 
    %         load(datafile,'X','fs','eogchannels','badchannels');
    %         for i=1:size(X,3), badchannels{i} = -1; end
    % standard eeg preprocessing (see function below). 
    %  Will discard EOG channels 
    
    

    %READ index 
    %data = readtable('new_selected_channels.csv');
    % Extract the ChannelsNumber column and increment each index by 1
    %channels_index = data.ChannelsNumber + 1;
    %channels_index = channels_index(1:69);
    
    X = X(:,:,:);
    %X = preprocess(X,fs);
    
    
    % T samples, D channels, N subjects
    [T,D,N] = size(X);  
    
    % now start the ISC code proper
    
    % compute cross-covariance between all subjects i and j
    Rij = permute(reshape(cov(X(:,:)),[D N  D N]),[1 3 2 4]); 
    
    % compute within- and between-subject covariances
    Rw =       1/N* sum(Rij(:,:,1:N+1:N*N),3);  % pooled over all subjects
    Rb = 1/(N-1)/N*(sum(Rij(:,:,:),3) - N*Rw);  % pooled over all pairs of subjects
    
    % shrinkage regularization of Rw
    Rw_reg = (1-gamma)*Rw + gamma*mean(eig(Rw))*eye(size(Rw));
    
    % +++ If multiple stimuli are available, then Rw and Rb should be averaged over
    % stimuli here prior to computing W and A +++
    
    % compute correlated components W using regularized Rw, sort components by ISC
    [W,ISC]=eig(Rb,Rw_reg); 
    [ISC,indx]=sort(diag(ISC),'descend'); W=W(:,indx);
    
    % compute forward model ("scalp projections") A
    A=Rw*W/(W'*Rw*W);
    
    % +++ If multiple stimuli are available, then Rij as computed for each stimulus
    % should be used in the following to compute ISC_persubject, and
    % ISC_persecond +++
    
    % Compute ISC resolved by subject, see Cohen et al.
    for i=1:N
        Rw=0; for j=1:N, if i~=j, Rw = Rw+1/(N-1)*(Rij(:,:,i,i)+Rij(:,:,j,j)); end; end
        Rb=0; for j=1:N, if i~=j, Rb = Rb+1/(N-1)*(Rij(:,:,i,j)+Rij(:,:,j,i)); end; end
        ISC_persubject(:,i) = diag(W'*Rb*W)./diag(W'*Rw*W);
    end
    
    % Compute ISC resolved in time
    for t = 1:floor((T-Nsec*fs)/fs)
        Xt = X((1:Nsec*fs)+(t-1)*fs,:,:);
        Rij = permute(reshape(cov(Xt(:,:)),[D N  D N]),[1 3 2 4]);
        Rw =       1/N* sum(Rij(:,:,1:N+1:N*N),3);  % pooled over all subjects
        Rb = 1/(N-1)/N*(sum(Rij(:,:,:),3) - N*Rw);  % pooled over all pairs of subjects
        ISC_persecond(:,t) = diag(W'*Rb*W)./diag(W'*Rw*W);
    end
    
    % show some results
    if ~exist('topoplot') | ~exist('notBoxPlot')
        warning('Get display functions topoplot, notBoxPlot where you found this file or on the web');
    else
            for i=1:Ncomp
                subplot(2,Ncomp,i);
                %topoplot(A(:,i),'BioSemi64.loc','electrodes','off'); title(['a_' num2str(i)])
            end
        subplot(2,2,3); notBoxPlot(ISC_persubject(1:Ncomp,:)'); xlabel('Component'); ylabel('ISC'); title('Per subjects');
        subplot(2,2,4); plot(ISC_persecond(1:Ncomp,:)'); xlabel('Time (s)'); ylabel('ISC'); title('Per second');
    end
    
    % ### Run all the code above with phase-randomized Xr to get chance values
    % of ISC measures under the null hypothesis of no ISC.
    Xr = phaserandomized(X);




% -------------------------------------------------------------------------
function X = preprocess(X,fs)
    % All the usual EEG preprocessing, except epoching and epoch rejection as
    % there are not events to epoch for natural stimuli. duh! Instead, bad data
    % is set = 0 in the continuous stream, which makes sense when computing
    % covariance matrices but maybe not for other purposes. Bad channels are
    % removed for all the indice given in badchannels cell array (each subject
    % has its own vector of indice). None are removed if this is set to []. If
    % it is set to -1, channels are removed based on outlies in power.
    
    debug = 0;     % turn this on to show data before/after preprocessing. 
    kIQD=4;        % multiple of interquartile differences to mark as outliers samples
    kIQDp=3;       % multiple of interquartile differences to mark as outliers channels
    HPcutoff = 0.5; % HP filter cut-off frequequency in Hz
    
    % pick your preferred high-pass filter
    [z,p,k]=butter(5,HPcutoff/fs*2,'high'); sos = zp2sos(z,p,k);
    
    [T,D,N]=size(X); 
    
    % if it is not EOG, then it must be EEG channel
    %eegchannels = setdiff(1:D,eogchannels);
    
    % Preprocess data for all N subjects
    for i=1:N
        
        data = X(:,:,i);
    
        % remove starting offset to avoid filter trancient
        data = data-repmat(data(1,:),T,1);
        
        % show the original data
        if debug, subplot(2,1,1); imagesc((1:T)/fs,1:D,data'); title(['Subject ' num2str(i)]); end
    
        % high-pass filter
        data = sosfilt(sos,data);          
        
        % regress out eye-movements;
        %data = data - data(:,eogchannels) * (data(:,eogchannels)\data);     
    
        % detect outliers above stdThresh per channel; 
        data(abs(data)>kIQD*repmat(diff(prctile(data,[25 75])),[T 1])) = NaN;
        
        % remove 40ms before and after;
        h=[1; zeros(round(0.04*fs)-1,1)];    
        data = filter(h,1,flipud(filter(h,1,flipud(data))));
        
        % Mark outliers as 0, to avoid NaN coding and to discount noisy channels
        data(isnan(data))=0;
    
        % Find bad channels based on power ourliers, if not specified "by hand"
        
        %logpower = log(std(data)); Q=prctile(log(std(data(:,:))),[25 50 75]);
        %badchannels{i} = find(logpower-Q(2)>kIQDp*(Q(3)-Q(1)));  
    
        
        % zero out bad channels
        %data(:,badchannels{i})=0; 
        
        % show the result of all this
        if debug, subplot(2,1,2); imagesc((1:T)/fs,1:D,data'); caxis([-100 100]); xlabel('Time (s)'); drawnow; end
    
        X(:,:,i) = data;
        
    end

% remove the eog channels as we have used them already
%X = X(:,eegchannels,:);


% -------------------------------------------------------------------------
function Xr = phaserandomized(X);
% Generate phase randomized surrogate data Xr that preserves spatial and
% temporal correlation in X, following Prichard D, Theiler J. Generating 
% surrogate data for time series with several simultaneously measured 
% variables. Physical review letters. 1994 Aug 15;73(7):951.

[T,D,N] = size(X);

Tr = round(T/2)*2; % this code only works if T is even; make it so
for i = 1:N
    Xfft = fft(X(:,:,i),Tr); % will add a zero at the end if uneven length
    Amp = abs  (Xfft(1:Tr/2+1,:)); % original amplitude
    Phi = angle(Xfft(1:Tr/2+1,:)); % orignal phase
    Phir = 4*acos(0)*rand(Tr/2-1,1)-2*acos(0); % random phase to add
    tmp(2:Tr/2,:) = Amp(2:Tr/2,:).*exp(sqrt(-1)*(Phi(2:Tr/2,:)+repmat(Phir,1,D))); % Theiler's magic
    tmp = ifft([Xfft(1,:); tmp(2:Tr/2,:); Xfft(Tr/2+1,:); conj(tmp(Tr/2:-1:2,:))]); % resynthsized keeping it real
    Xr(:,:,i) = tmp(1:T,:,:); % grab only the original length
end


