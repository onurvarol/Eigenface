classdef FaceRecognizer < handle
    %FACERECOGNÝZER Class of facerecognition operations
    %   - In this class initialization of the training and testing sets are
    %   loaded.
    %   - Facerecognizer trains and analysis is carried out 
    properties (Access = 'private')
        trainPath
        testPath
        K
        eigValues
        eigVectors
        bestM
    end
    properties
        trainData
        testData
        meanFace
    end
    
    methods
        % Constructor
        function obj = FaceRecognizer(testPath, trainPath) 
            obj.testPath = testPath;
            obj.trainPath = trainPath;
        end
        
        % Trains the face recognizer
        function obj = Train(obj)
            obj.trainData = obj.loadImages(obj.trainPath, 'jpg');
            obj.testData = obj.loadImages(obj.testPath, 'jpg');
            m = zeros(size(obj.trainData,1));
            for i=1:size(obj.trainData,2)
                X(i,:) = obj.trainData{i}.vector;
                m = m + obj.trainData{i}.vector;
            end
            m = m / size(obj.trainData,2);
            for i=1:size(obj.trainData,2)
                X(i,:) = X(i,:) - m; % mean is substracted   
            end
            obj.meanFace = m;
            
            obj.K = (X*X')/size(obj.trainData,2); % KxK matrix

            [V,D] = eig(obj.K); % Eigenvalues and eigenvectors

            for i=1:size(D,1)
               eigVal(i) = D(i,i); % Get vector of diagonal elements
            end

            [B, IX] = sort(eigVal,'descend');
            obj.eigValues = B;
            for j = 1:size(V,2)
                eigVec(:,j) = V(:,IX(:,j)); % sort columns of the eigenvalue matrix
            end
            
            for i =1:size(eigVec,2)
               v = X'*eigVec(:,i);
               obj.eigVectors(:,i) = v/norm(v); % Vk reduced eigenvector
            end
            
            %% M values calculates
            thres = 0.95;
            sumV = sum(obj.eigValues);
            sumT = 0;
            M = 1;
            while(sumT < sumV*thres)
               sumT = sumT + obj.eigValues(M);
               M = M + 1;
            end
            
            %M = 58; % Cheating in here =)
            obj.bestM = M;
            %% Trainin set features create
            feature = zeros(1,M);
            for i = 1:size(obj.trainData,2) % Creation of the training images feature vectors
                for j=1:size(obj.trainData,2)
                   feature(j) = dot(obj.trainData{i}.vector-obj.meanFace,obj.eigVectors(:,j)); % bm values
                end
                obj.trainData{i}.features = feature;
            end
            
            %% Test image feature extraction
           for i=1:size(obj.testData,2)
               testVec = obj.testData{i}.vector - obj.meanFace;

               for j=1:size(obj.trainData,2)
                   feature(j) = dot(testVec,obj.eigVectors(:,j)); % bm values
               end
               obj.testData{i}.features = feature;
           end
        end
        
        % Makes test on already defined image or load another
        % index : index of the test images
        % bestN : displays best N result      
        function obj = Recognize(obj,index, bestN)
            if(index > 0)
               % Apply on i^th test image  
               testVec = obj.testData{index}.vector - obj.meanFace;
                
               for j=1:size(obj.trainData,2)
                   feature(j) = dot(testVec,obj.eigVectors(:,j)); % bm values
               end
               obj.testData{index}.features = feature;
               
               img = zeros(length(testVec),1);
               for j=1:obj.bestM
                   img = img + (obj.testData{index}.features(j))*obj.eigVectors(:,j); % reconstraction of the minimized face
               end
               figure;
               subplot(2,bestN+1,1)
               imagesc(obj.vector2img(img,62)); % eigenface of test data
               colormap gray; axis off
               title('Eigen Image')
               
               subplot(2,bestN+1,bestN+2)
               imshow(obj.testData{index}.image); % eigenface of test data
               colormap gray; axis off
               title('Original Image')
               
               %% Recognize bestN image
               
               for j = 1:size(obj.trainData,2)
                    sumb(j) = 0;
                    for k=1:obj.bestM
                        sumb(j) = sumb(j) + ((obj.trainData{j}.features(k)-obj.testData{index}.features(k))^2)/obj.eigValues(k);
                    end
               end
                
               for b=1:bestN
                    [B, IX] = sort(sumb);
                    subplot(2,bestN+1,[b+1 bestN+b+2]);
                    imshow(obj.trainData{IX(b)}.image);
                    colormap gray; axis off
                    title(['Match-' num2str(b)])
               end
            else
               % Load from file
               
            end
            
        end
        
        % Analysis best M value for different comparison conditions
        function obj = AnalyseBestM(obj, realIndex)
            errors = zeros(size(obj.trainData,2)-1,length(realIndex));
            for i=1:size(obj.trainData,2)-1
                for j=2:length(realIndex)
                    for k=1:i
                        %errors(i,j) = errors(i,j) + ((obj.trainData{realIndex(j)}.features(k)-obj.testData{j}.features(k))^2)/(obj.eigValues(k)*sqrt(i));
                        %errors(i,j) = errors(i,j) + ((obj.trainData{realIndex(j)}.features(k)-obj.testData{j}.features(k))^2)/(obj.eigValues(k)*i);
                        errors(i,j) = errors(i,j) + ((obj.trainData{realIndex(j)}.features(k)-obj.testData{j}.features(k))^2);%;/(obj.eigValues(k));
                    end
%                   errors(i,j) = acos(dot(obj.trainData{realIndex(j)}.features(1:i),obj.testData{j}.features(1:i))/...
%                         (norm(obj.trainData{realIndex(j)}.features(1:i))*norm(obj.testData{j}.features(1:i))));
                end
            end
            figure;
            find(sum(errors,2) == min(sum(errors,2)))
            plot(sum(errors,2))
            title('Errors in different M')
            xlabel('M values')
            ylabel('Errors')
        end
        
        % Plots training set images
        function obj = PlotTrainSet(obj)
            figure; % Display training set
            for i=1:size(obj.trainData,2)
               subplot(8,10,i)
               imagesc(obj.trainData{i}.image)
               colormap gray
               title(['ID: ' num2str(i)])
               axis off
            end
        end
        
        % Plots N eigenface
        function obj = PlotEigenfaces(obj,N)
           figure;
           for i=1:N
              subplot(1,N,i);
              imagesc(obj.vector2img(obj.eigVectors(:,i), 62))
              colormap gray
              title(['Eigenface: ' num2str(i)])
              axis off
           end
        end
        
        % Motivation and fun stuff =)
        function obj = Hallelujah(obj,str)
            fprintf([str '\n']);
            load handel
            fs=10000;
            sound(y,fs)
        end
    end
    
    methods (Access = 'private')
        % Loads dataset of train and test sets
        function data = loadImages(obj,path, ext)
            files = dir([path '/*.' ext]);
            for i=1:length(files)
               temp = files(i);
               data{i}.name = temp.name;
               data{i}.image = im2double(imread([path '\' temp.name]));
               % data{i}.image = im2double(imadjust(imread([path '\' temp.name])));
               data{i}.vector = [];
               for j=1:size(data{i}.image,1)
                  data{i}.vector = [data{i}.vector data{i}.image(j,:)]; 
               end
               data{i}.features = [];
               if(strcmp(path,'test'))
                  data{i}.realIndex = -1; 
               end
            end
        end
        
        % Coýnverts the image vectors to image
        function img = vector2img(obj,vector, h)
            w = length(vector)/h;
            for i=1:h
               img(i,:) = vector((i-1)*w+1:i*w); 
            end
        end
    end
    
end

