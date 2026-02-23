function [ISC, ISC_persubject, W, A] = multiStimISC(data, refsubjects, fs)
% data is a cell of volumes of dimensiosn T*D*N, where D must be the same for all volumes
% refsubjects is cell of arrays with the reference subjects for each stimulus
%           --- This list should include the indices in N for the reference subjects (subjects which all others will be compared to)
% fs is the recording sampling rate
% the two lists must have the same length corresponding to the number of stimuli

% some ISC processing parameters
gamma = 0.1; % shrinkage parameter; smaller gamma for less regularization
Nsec  = 5;  % time-window (in seconds) over which to compute time-reposeved ISC
Ncomp = 3;  % number of components to dispaly (all D are computed)
Nstim = length(data);

if length(data)~=length(refsubjects)
  error('All lists must have the same number of subjects in it.')
end

Dcommon = size(data{1},2);

% compute within- and between-subjects covrariances averaged across all stimuli
Rw = 0;
for s=1:Nstim

  X = data{s};

  [T,D,N{s}] = size(X);

  if D~=Dcommon, error('All datasets must have same number of channels!'); end

  Rij{s} = permute(reshape(cov(X(:,:)),[D N{s}  D N{s}]),[1 3 2 4]); % check this line!!!

  % compute within- and between-subject covariances
  Rw = Rw +       sum(Rij{s}(:,:,1:N{s}+1:N{s}*N{s}),3)/Nstim;  % pooled over all subjects

end

Rb = 0;
for s=1:Nstim

  X = data{s};


  [T,D,N{s}] = size(X);

  if D~=Dcommon, error('All datasets must have same number of channels!'); end

  Rij{s} = permute(reshape(cov(X(:,:)),[D N{s}  D N{s}]),[1 3 2 4]); % check this line!!!

  % compute within- and between-subject covariances
  Rb = Rb + 1/(N{s}-1)*(sum(Rij{s}(:,:,:),3) - Rw);  % pooled over all pairs of subjects

end

Rb = Rb/Nstim;

% shrinkage regularization of Rw
Rw_reg = (1-gamma)*Rw + gamma*mean(eig(Rw))*eye(size(Rw));

% compute correlated components W using regularized Rw, sort components by ISC
[W,ISC]=eig(Rb,Rw_reg); [ISC,indx]=sort(diag(ISC),'descend'); W=W(:,indx);

% compute forward model ("scalp projections") A
A=Rw*W*inv(W'*Rw*W);

% Compute ISC resolved by stimulus
for s=1:Nstim

  % Compute ISC resolved by subject, comparing only with reference subject
  for i=1:N{s}

      Rw=0; for j=refsubjects{s}, if i~=j, Rw = Rw+(Rij{s}(:,:,i,i)+Rij{s}(:,:,j,j)); end; end
      Rb=0; for j=refsubjects{s}, if i~=j, Rb = Rb+(Rij{s}(:,:,i,j)+Rij{s}(:,:,j,i)); end; end
      ISC_persubject{s}(:,i) = diag(W'*Rb*W)./diag(W'*Rw*W);

  end

end
