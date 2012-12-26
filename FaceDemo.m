clear all;

facereco = FaceRecognizer('test', 'training'); % Defines the training and test sets
facereco.Train(); % Train the facerecognizer
facereco.PlotTrainSet();
facereco.PlotEigenfaces(5); % Plots most significant N eigenvector
facereco.Recognize(1,3); % Recognizes i^th test image and shows N best match
facereco.AnalyseBestM([68 68 68]); % Analysis best M for given test set

%facereco.Hallelujah('Fun time =)');


