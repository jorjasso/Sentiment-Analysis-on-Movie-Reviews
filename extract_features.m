function [S, Var1]=extract_features(listOfWords,feature_word)
%reading Test set
delimiter = '\t';
startRow = 2;
%formatSpec = '%n%n%n%n%n%n%n%n%s%[^\n\r]';
formatSpec = '%n%n%s%n';
fileID = fopen('test.tsv','r');
C = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);

%variables
Var1=C{1,1};%PhraseId
Var2=C{1,2};%SentenceId
Var3=C{1,3};%Phrase
%phraseID=Var1;
N=length(Var1);
clear C delimiter formatSpec  Var2  

Var3=cellfun(@lower, Var3,'UniformOutput',0); % to lower case letters
sentence =cellfun(@strsplit, Var3,'UniformOutput',0); % split sentences into words
nWordsBySentence=cellfun(@length, sentence); % counting the number of words per sentences
%listOfWords=unique([sentence{:}]); %a dictionary of the words in the training set
numberOfClass=5;

% grouping the sentences by class


%load featureI.mat % load listOfWords and features
%load exp.mat

S=cell(1,N);
for i=1:N   
    idx=[];
    for w=1:length(sentence{i})
        word=sentence{i}(w);
        idx=[idx,find(cell2mat(cellfun(@(x) sum(strcmp(x,word)),listOfWords, 'UniformOutput',0))==1)];
    end
    %each sentence is a distribution of points
    S{i}=feature_word(idx,:);    
end

