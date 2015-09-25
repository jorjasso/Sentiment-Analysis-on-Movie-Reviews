function G=kernelPM(S, SS,kernelParam, option)
% kernel on probability measures
% % References: Support measure data description support measure description for group anomaly detection
% input
%     S      = trainig set of replicates: S={S{1},S{2},...,S{N}}. Each S{i} has L replicates for observartion i
%     STest	     = test set of replicates
%     kernelOp = 1 lineal, 2 = polinomyal, 3 = RBF kernel, 4 = RBF kernel with spherical normalization
% output
% 	Krr = kernel matrix between training samples S
% 	Kre = kernel matrix between training samples and test samples
%	Kee = kernel matrix between test samples
%       K_traceTraining = trace of the covariance operator in the RKHS of training set
%	K_traceTest = trace of the covariance operator in the RKHS of test set
% jorge.jorjasso@gmail.com


%----------------------
n=length(S);
m=length(SS);
G=zeros(n,m);
switch option
    case 0 % slow, first version
        for i=1:n
            for j=1:m
                [L,~]=size(S{i});
                [LL,~]=size(SS{j});
                k=exp(-0.5*kernelParam*sqdistAll(S{i},SS{j}));
                kVal=sum(sum(k));%
                G(i,j)= kVal/(L*LL);
            end
        end
    case 1 %spherical normalization
        Kii=cellfun(@(x) sum(sum(exp(-0.5*kernelParam*sqdistAll(x,x)))),S ,'UniformOutput', false );
        L_vec = cell2mat(cellfun(@(x) size(x,1), S ,'UniformOutput', false));
        Kii=cell2mat(Kii)./(L_vec.^2);
        
        
        Kjj=cellfun(@(x) sum(sum(exp(-0.5*kernelParam*sqdistAll(x,x)))),SS ,'UniformOutput', false );
        L_vec = cell2mat(cellfun(@(x) size(x,1), SS ,'UniformOutput', false));
        Kjj=cell2mat(Kjj)./(L_vec.^2);
        %------      
        L=cell2mat(cellfun(@(x) size(x,1), S ,'UniformOutput', false));
        LL=cell2mat(cellfun(@(x) size(x,1), SS ,'UniformOutput', false));
        for i=1:n
            S_i=repmat(S(i),m,1);
            Li= repmat(L(i),m,1);                      
            Gi=cellfun(@(x,y) sum(sum(exp(-0.5*kernelParam*sqdistAll(x,y)))),S_i,SS ,'UniformOutput', false );
            Gi=cell2mat(Gi)./(Li.*LL);
            G(i,:)=Gi;
        end
        %---------
        G=G./sqrt(Kii'*Kjj);
    case 2 % faster
        L=cell2mat(cellfun(@(x) size(x,1), S ,'UniformOutput', false));
        LL=cell2mat(cellfun(@(x) size(x,1), SS ,'UniformOutput', false));
        for i=1:n
            S_i=repmat(S(i),1,m);
            Li= repmat(L(i),1,m);                      
            Gi=cellfun(@(x,y) sum(sum(exp(-0.5*kernelParam*sqdistAll(x,y)))),S_i,SS ,'UniformOutput', false );
            Gi=cell2mat(Gi)./(Li.*LL);
            G(i,:)=Gi;
        end
end
