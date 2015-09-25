function batchTraining(i, partition,machine)
%%
addpath ./MAT/
disp('Reading Test set')
load readData.mat
clearvars -except Var4 partition i machine
N=length(Var4);
y=Var4;
load CVParammeteres.mat
load features.mat
clear feature_word listOfWords Var4
addpath ./SVM-KM/
ind=randperm(N,floor(N/partition));

% Batch Training
%------------------
kernelOption=1;
label=Y(i,:)';
disp('training: i, partition')
[i,partition]
[b,alph1,pos, nSV, Err_Rate, Err_RateA,Err_RateN,~,AUC,~]=getStatisticsSVM(S',label,ind,ind,kernelOption, bestGamma,bestC);
fileName=strcat(int2str(i),'model',machine,'.mat');
disp('finished')
save(fileName,'b','alph1', 'pos');
exit

