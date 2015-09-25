% jorge.jorjasso@gmail.com,
% modified version of getStatisticsSVM.m in TSK-kernels-Low-quality
%  at
%  https://github.com/jorjasso/TSK-kernels-Low-quality/blob/master/getStatisticsSVM.m
% date 09 set 2015
function [b,alph1,pos, nSV, Err_Rate, Err_RateA,Err_RateN,ConfMat,AUC,stat]=getStatisticsSVM(Training,label,indTraining,indValidation,kernelOption, kernelParam,C)


X=Training(indTraining',:);
Z=Training(indValidation,:);

y=label(indTraining);
yV=label(indValidation);
G=kernelPM(X, X,kernelParam,kernelOption);

% Train a SVM
[b,alph1,pos]=SVM_MONQP(G,y,C);

clear G

V=kernelPM(X(pos,:), Z,kernelParam,kernelOption);

clear X Z

% Test
ypred = V'*(y(pos).*alph1) + b;
% statistics = compare ypred agains yV
nSV=length(pos);
[Err_Rate, Err_RateA,Err_RateN,ConfMat,AUC,stat]=computeError(yV,ypred);


%--------------------------------
%TestCase
% Example
% n = 500; % upto n = 10000;
% sigma=1.4;
% [Xapp,yapp,Xtest,ytest]=dataset_KM('checkers',n,n^2,sigma);
% kernelOption=3; kernelParam=0.5;
% G=getKernel(kernelOption,X,X,kernelParam);
% [b,alph1,pos]=SVM_MONQP(G,y,C)
% V=getKernel(kernelOption,Z,X(pos,:),kernelParam);
% ypred = V*(y(pos).*alph1) + b;
% [Err_Rate, Err_RateA,Err_RateN,ConfMat,AUC,stat]=computeError(yV,ypred);
