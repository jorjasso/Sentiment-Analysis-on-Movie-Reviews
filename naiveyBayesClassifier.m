
%% Read data set
addpath ./MAT/
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

save ./MAT/preprocessing.mat sentence listOfWords nWordsBySentence numberOfClass sentencesClass nWordClass priorClass
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

save ./MAT/feature_word.mat feature_word
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

%% Naive Bayes Classifier
addpath ./MAT/
disp('Classifier')
load readData.mat
load preprocessing.mat
N=length(Var4);
load features.mat
%[Z,phraseID]=extract_features(listOfWords,feature_word)
load testFeature.mat

ypred=cell(1,5);
likelihood=cellfun(@(x) sum(log(x),1),Z,'UniformOutput',0);
for i=1:5   
    i
    logL=cellfun(@(x) x(i), likelihood);
    ypred{i}=(log(priorClass(i))+  logL)';
    size(logL)
    size(ypred)
end
size(phraseID)

[~,prediction]=max(cell2mat(ypred)');

M=[phraseID,prediction'];
dlmwrite('submissionBayes.csv',M,'precision',6,'roffset',1);   
exit

