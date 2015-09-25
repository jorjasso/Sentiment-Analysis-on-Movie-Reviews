
%% Read data set
delimiter = '\t';
startRow = 2;
%formatSpec = '%n%n%n%n%n%n%n%n%s%[^\n\r]';
formatSpec = '%n%n%s%n';
fileID = fopen('train.tsv','r');
C = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);

%variables
Var1=C{1,1};%PhraseId
Var2=C{1,2};%SentenceId
Var3=C{1,3};%Phrase
Var4=C{1,4};%Sentiment

clear C delimiter formatSpec
save ./MAT/readData.mat Var1 Var2 Var3 Var4

%% preprocessing
addpath ./MAT/
disp('preprocessing')
load readData.mat

Var3=cellfun(@lower, Var3,'UniformOutput',0); % to lower case letters
sentence =cellfun(@strsplit, Var3,'UniformOutput',0); % split sentences into words
nWordsBySentence=cellfun(@length, sentence); % counting the number of words per sentences
listOfWords=unique([sentence{:}]); %a dictionary of the words in the training set
numberOfClass=5;

% grouping the sentences by class
N=length(Var1);
[phraseByClass, I]=sortrows([Var1,Var4],2);


%computing priors: #sentencesByClass/TotalSentences
priorClass=zeros(1,numberOfClass);
for i=1:numberOfClass
    priorClass(i)=length(find(phraseByClass(:,2)==(i-1)))/N; % number of sentences by class/ total of sentences
end

%computing the indices of sentences by class
ind=cell(1,numberOfClass);
for i=1:numberOfClass
    ind{i}=(Var4==(i-1));
end

% smaller dictionaries of words per class
listOfWordsClass=cell(1,numberOfClass);
for i=1:numberOfClass
    listOfWordsClass{i}=unique([sentence{ind{i}}]);
end

%sentences per class
sentencesClass=cell(1,numberOfClass);
for i=1:numberOfClass
    sentencesClass{i}=sentence(ind{i});
end

% number of unique words per class
nWordClass=cell(1,numberOfClass);
for i=1:numberOfClass
    nWordClass{i}=length(listOfWordsClass{i});
end

save ./MAT/preprocessing.mat sentence listOfWords nWordsBySentence numberOfClass sentencesClass nWordClass
clearvars -except  sentence listOfWords nWordsBySentence numberOfClass sentencesClass nWordClass

%% feature extraction
addpath ./MAT/

disp('feature extraction')
load preprocessing.mat
% each word w has five featureswho
% given by feature_word=[prob(word_w/class1), ..., prob(word_w/classN)]
f=zeros(1,5); %for the features
word_frequency=zeros(1,5);
numberFeatures=5;
feature_word=zeros(length(listOfWords),numberFeatures);
B=length(listOfWords);
for w=1:length(listOfWords)% for each word in list of words
    word=listOfWords{w};
    
    %number of word per class
    for i=1:numberOfClass
        word_frequency(i)=sum(cell2mat(cellfun(@(x) sum(strcmp(x,word)),sentencesClass{i}, 'UniformOutput',0)));
        % frequency of word per class + Laplacian smoothibng
        f(i)=(word_frequency(i)+1)/(nWordClass{i}+B);
    end
    
    % feature_word=[prob(word1/class1), ..., prob(word1/classN)] %likelihood
    feature_word(w,:)=f(:);
    
end

save feature_word.mat feature_word
% each sentence has a set of features given the agreggation of the
% features of the words (feature) occuring in the sentence
S=cell(1,N);
for i=1:N
    idx=[];
    for w=1:length(sentence{i})
        word=sentence{i}(w);
        idx=[idx,find(cell2mat(cellfun(@(x) sum(strcmp(x,word)),listOfWords, 'UniformOutput',0))==1)]; % indice of word w in list of words
    end
    %each sentence is a distribution of points
    S{i}=feature_word(idx,:);
    
end

save ./MAT/features.mat S feature_word listOfWords
save ./MAT/allVariables.mat
clear sentence feature word word_frequency f nWordClass sentencesClass listOfWordsClass ind priorClass phraseByClass listOfWords

%% SVM Classifier
addpath ./SVM-KM/
addpath ./MAT/

disp('Classifier')
load readData.mat
clearvars -except Var4
N=length(Var4);
load features.mat
% As each observation (sentence ) is given by a set of features (S{i}), we use a
% multiclass support vector machine with kernel on distributions (see kernelPM.m and SVM_MONQP.m);

%Train a one-versus all SVM on probability measures (on distributions)

%labels
y=Var4;
y1(y==0)=1; y1(y~=0)=-1;
y2(y==1)=1; y2(y~=1)=-1;
y3(y==2)=1; y3(y~=2)=-1;
y4(y==3)=1; y4(y~=3)=-1;
y5(y==4)=1; y5(y~=4)=-1;

Y=[y1;y2;y3;y4;y5];
clear y1 y2 y3 y4 y5

% compute grid for the kernel paramenter using the "mean heuristic"
tamSmallData=1000;
ind =randperm(N,tamSmallData);
smallX=S(ind);
X=cell2mat(smallX');
M=sqdistAll(X,X);
quantiles=quantile(M(:),[0.1, 0.5, 0.9]);
grid_gamma=[1/quantiles(1),1/quantiles(2),1/quantiles(3)];
grid_C=[] % the grid for C is constructed inside gridSearch.m in smart way

% Model selection with cross validation and coarse grid search
% As the data set is unbalanced the best parameters are found looking  the
% AUC values (see gridSearch.m and getStatisticsSVM.m)


C=[];
gamma=[];
list_AUC=[];
list_ACC=[];
tamSmallData=500;

% cross validation indices
CVPCell=cell(5,1);
indCell=cell(5,1);
for i=1:numberOfClass
    indCell{i} =randperm(N,tamSmallData);
    CVPCell{i}=cvpartition(Y(i,indCell{i}),'k',5);
end

% cross valitation and multiclass SVM using the one-vs-all strategy
for i=1:numberOfClass
    i
    smallX=S(indCell{i});
    [bestC,bestGamma, accuracy, ~, ~, ~,~,AUC]=gridSearch(CVPCell{i}, smallX, Y(i,indCell{i})',[], grid_C, grid_gamma);
    C=[C,bestC];
    gamma=[gamma,bestGamma];
    list_AUC=[list_AUC, AUC];
    list_ACC=[list_ACC,accuracy];
end
% Looking the AUC value.
[~ , I]=max(list_AUC);
bestGamma=gamma(I);
bestC=C(I);
save ./MAT/CVParammeteres.mat  bestGamma bestC Y
%bestGamma=1/quantiles(2);bestC=1;
% Train and test with the best cross validation parammeters (bestGamma, bestC)
%% Training + Testing
addpath ./MAT/
disp('Reading Test set')
load readData.mat
clearvars -except Var4
N=length(Var4);
y=Var4;
load CVParammeteres.mat
load features.mat
clear feature_word listOfWords Var4
addpath ./SVM-KM/

%model
bCell=cell(1,5); alphCell=cell(1,5); posCell=cell(1,5);

disp('Training and Testing')

% Training
kernelOption=1% 1 RBF kernel on pm with spherical normalization, 1=RBF kernel on pm, 2
for i=1:5
    i
    ind=randperm(N,floor(N/100));
    label=Y(i,:)';
    [b,alph1,pos, nSV, Err_Rate, Err_RateA,Err_RateN,~,AUC,~]=getStatisticsSVM(S',label,ind,ind,kernelOption, bestGamma,bestC);
    bCell{i}=b; alphCell{i}=alph1; posCell{i}=pos;
    
end

%save ./MAT/model.mat bCell alphCell posCell


% Testing
% feature extraction of the test set
[Z,phraseID]=extract_features(listOfWords,feature_word)
%save ./MAT/testFeature.mat Z
load testFeature.mat
load modelTolkein.mat
ypred=cell(1,5);
for i=1:5
    V=kernelPM(S(:,posCell{i}), Z,bestGamma);
    ypred{i} = V'*(y(posCell{i}).*alphCell{i}) + bCell{i};
end
[~,prediction]=max(cell2mat(ypred)');

M=[phraseID,prediction'];

dlmwrite('./submisionCSV/submissionModelTolkien.csv',M,'precision',6,'roffset',1);   

