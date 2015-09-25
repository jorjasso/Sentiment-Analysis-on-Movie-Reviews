function batchTest(iter,model)
addpath ./MAT/
disp('Reading Test set')
load readData.mat
%clearvars -except Var4
N=length(Var4);
y=Var4;
load CVParammeteres.mat
load features.mat
clear feature_word listOfWords Var4
addpath ./SVM-KM/
load testFeature.mat
load(model)
ypred=cell(1,5);

%val=1:5000:65000;
val=1:10:100;
kernelOption=2
if iter<13
    ZZ=Z(val(iter):val(iter+1)-1);
    phraseID=phraseID(val(iter):val(iter+1)-1);
else
    ZZ=Z(val(iter):end);
    phraseID=phraseID(val(iter):end);
end

for i=1:5
    i
    V=kernelPM(S(:,posCell{i}), ZZ,bestGamma,kernelOption);
    ypred{i} = V'*(y(posCell{i}).*alphCell{i}) + bCell{i};
end
[~,prediction]=max(cell2mat(ypred)');

M=[phraseID,prediction'];

name=strcat(int2str(iter),'submission',model,'.csv');
dlmwrite(name,M,'precision',6);
exit
