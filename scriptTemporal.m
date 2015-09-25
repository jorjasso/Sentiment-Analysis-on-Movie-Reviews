%% batch training running in Gauss, spherical normalization kernel, N/50
 
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(1, 50,'gauss')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(2, 50,'gauss')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(3, 50,'gauss')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(4, 50,'gauss')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(5, 50,'gauss')";
 
 % batch training running in proteina, spherical normalization kernel, N/10
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(1, 10,'proteina')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(2, 10,'proteina')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(3, 10,'proteina')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(4, 10,'proteina')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(5, 10,'proteina')";

 
% load models batch training
bCell=cell(1,5); alphCell=cell(1,5); posCell=cell(1,5);

for i=1:5
    fileName=strcat(int2str(i),'modelTolstoi.mat');
    load(fileName)
    b
    bCell{i}=b; alphCell{i}=alph1; posCell{i}=pos;
end

save modelTolkein.mat bCell alphCell posCell

% usar opcion -nojvmn y -nodesktop

%% batch testing
% Testing in tolkien (termino y submeti)
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(1, 'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(2,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(3,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(4,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(5,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(6,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(7,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(8,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(9,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(10,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(11,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(12,'modelTolkein')";
screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTest(13,'modelTolkein')";


%% concatenate several cvs files

files = dir('*s.csv');  % Get list of files
out = csvread(files(1).name);  % First file
for ii = 2:numel(files)
    new = csvread(files(ii).name);  % Read the nth file
    out = vertcat(out, new);   % Concatenate "out" with the 2nd column from new file
end

dlmwrite('submissionModelGauss.csv',out,'precision',6,'roffset',1);

%% experiment info
% testing modelTolkien N/100 in tolkien
% testing modelGauss in inti N/???
% training modelGauss N/50 in Gauss
% training modelProteina N/10 in Proteina
% Naive bayes JorgeGuevaraDíaz 	0.18408 	
% SVM JorgeGuevaraDíaz 	0.19846 (N/100 submission model Tolkein)
% JorgeGuevaraDíaz 	0.19846 submissionModelGauss
% JorgeGuevaraDíaz 	0.19846 submissionModelTolkien
% http://www.nltk.org/
% http://scikit-learn.org/stable/
% https://github.com/rafacarrascosa/samr/blob/develop/README.md
%% batch training complete model
%% batch training running in Gauss, spherical normalization kernel
 
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(1, 1,'tolstoiComplete')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(2, 1,'dostoievskiComplete')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(3, 1,'puchkinComplete')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(4, 1,'gauss')";
 screen  -d -m matlab -nodisplay -nosplash --nojvmn -r     "batchTraining(5, 1,'gauss')";
 