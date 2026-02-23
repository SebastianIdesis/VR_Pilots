function [ISC,ISC_persubject,ISC_persecond,W,A,p_ISC,p_ISC_persecond,null] = ISC_Null(datafile)

    gamma = 0.5;
    Nsec  = 15;
    Ncomp = 3;

    X = datafile;
    fs = 500;
    X = X(:,:,:);
    X = preprocess(X,fs);

    [T,D,N] = size(X);

    % ---- Compute observed ISC (your existing code) ----
    Rij = permute(reshape(cov(X(:,:)),[D N  D N]),[1 3 2 4]);

    Rw =       1/N* sum(Rij(:,:,1:N+1:N*N),3);
    Rb = 1/(N-1)/N*(sum(Rij(:,:,:),3) - N*Rw);

    Rw_reg = (1-gamma)*Rw + gamma*mean(eig(Rw))*eye(size(Rw));

    [W,ISCmat]=eig(Rb,Rw_reg);
    [ISC,indx]=sort(diag(ISCmat),'descend'); W=W(:,indx);

    A=Rw*W/(W'*Rw*W);

    % Per-subject ISC (observed)
    ISC_persubject = zeros(D,N);
    for i=1:N
        Rw_i=0; 
        for j=1:N
            if i~=j
                Rw_i = Rw_i+1/(N-1)*(Rij(:,:,i,i)+Rij(:,:,j,j));
            end
        end
        Rb_i=0; 
        for j=1:N
            if i~=j
                Rb_i = Rb_i+1/(N-1)*(Rij(:,:,i,j)+Rij(:,:,j,i));
            end
        end
        ISC_persubject(:,i) = diag(W'*Rb_i*W)./diag(W'*Rw_i*W);
    end

    % Per-second ISC (observed)
    nWin = floor((T-Nsec*fs)/fs);
    ISC_persecond = zeros(D,nWin);
    for t = 1:nWin
        Xt = X((1:Nsec*fs)+(t-1)*fs,:,:);
        Rij_t = permute(reshape(cov(Xt(:,:)),[D N  D N]),[1 3 2 4]);
        Rw_t =       1/N* sum(Rij_t(:,:,1:N+1:N*N),3);
        Rb_t = 1/(N-1)/N*(sum(Rij_t(:,:,:),3) - N*Rw_t);
        ISC_persecond(:,t) = diag(W'*Rb_t*W)./diag(W'*Rw_t*W);
    end

    % ---- Significance testing with phase-randomized surrogates ----
    Nsurr = 200;          % increase to 500-2000 for more stable p-values
    K = min(Ncomp, D);    % number of components to test

    null.ISC = zeros(K,Nsurr);
    null.max_ISC_persecond = zeros(K,Nsurr);  % for max-statistic across time
    % (optional) store full null timecourses (can be big): null.ISC_persecond = zeros(K,nWin,Nsurr);

    for s = 1:Nsurr
        Xr = phaserandomized(X);

        % recompute ISC for surrogate (recommended for eigenvalue-based ISC)
        Rij_r = permute(reshape(cov(Xr(:,:)),[D N  D N]),[1 3 2 4]);
        Rw_r =       1/N* sum(Rij_r(:,:,1:N+1:N*N),3);
        Rb_r = 1/(N-1)/N*(sum(Rij_r(:,:,:),3) - N*Rw_r);
        Rw_reg_r = (1-gamma)*Rw_r + gamma*mean(eig(Rw_r))*eye(size(Rw_r));

        [~,ISCmat_r]=eig(Rb_r,Rw_reg_r);
        ISC_r = sort(diag(ISCmat_r),'descend');
        null.ISC(:,s) = ISC_r(1:K);

        % time-resolved null using surrogate data:
        % compute ISC_persecond under null using the *surrogate* W (or recompute W per window—costly).
        % Here we use the global surrogate eigenvectors (consistent with observed approach).
        [Wr,~] = eig(Rb_r,Rw_reg_r);
        % sort Wr same as ISC_r order:
        [~,indr]=sort(diag(ISCmat_r),'descend'); Wr=Wr(:,indr);

        maxVals = -inf(K,1);
        for t = 1:nWin
            Xrt = Xr((1:Nsec*fs)+(t-1)*fs,:,:);
            Rij_rt = permute(reshape(cov(Xrt(:,:)),[D N  D N]),[1 3 2 4]);
            Rw_rt =       1/N* sum(Rij_rt(:,:,1:N+1:N*N),3);
            Rb_rt = 1/(N-1)/N*(sum(Rij_rt(:,:,:),3) - N*Rw_rt);
            isc_rt = diag(Wr'*Rb_rt*Wr)./diag(Wr'*Rw_rt*Wr);
            maxVals = max(maxVals, isc_rt(1:K));
            % null.ISC_persecond(:,t,s) = isc_rt(1:K); % optional
        end
        null.max_ISC_persecond(:,s) = maxVals;
    end

    % Component-wise p-values (one-sided: "greater than chance")
    p_ISC = zeros(K,1);
    for k = 1:K
        p_ISC(k) = (1 + sum(null.ISC(k,:) >= ISC(k))) / (1 + Nsurr);
    end

    % Per-second p-values with max-statistic FWER correction across time
    p_ISC_persecond = ones(K,nWin);
    for k = 1:K
        for t = 1:nWin
            obs = ISC_persecond(k,t);
            % compare to distribution of max over time under null:
            p_ISC_persecond(k,t) = (1 + sum(null.max_ISC_persecond(k,:) >= obs)) / (1 + Nsurr);
        end
    end

    % ---- optional displays (your existing block) ----
    if ~exist('topoplot','file') || ~exist('notBoxPlot','file')
        warning('Get display functions topoplot, notBoxPlot where you found this file or on the web');
    else
        subplot(2,2,3); notBoxPlot(ISC_persubject(1:K,:)'); xlabel('Component'); ylabel('ISC'); title('Per subjects');
        subplot(2,2,4); plot(ISC_persecond(1:K,:)'); xlabel('Time (s)'); ylabel('ISC'); title('Per second');
    end
end

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
end

% -------------------------------------------------------------------------
function Xr = phaserandomized(X)
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
end



