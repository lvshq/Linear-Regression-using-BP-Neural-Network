% clc
% clear
%%
% inputData = csvread('sample_trainV1.csv',1,1);
inputData = csvread('train.csv',1,1);
[rowIn, colIn] = size(inputData);
% disp(rowIn);
% disp(colIn);
% % RNG = [1, 385, 665, 385];
RNG1 = [1, colIn, rowIn, colIn];
% % outputDataTemp = csvread('sample_trainV1.csv',1,1, RNG1);
outputDataTemp = csvread('train.csv',1,1, RNG1);
[rowOut, colOut] = size(outputDataTemp); % m is row number, n is column number
% remove the 'id' column
outputData = outputDataTemp(2:rowOut, 2:colOut);
% remove the last 'reference' column (namely the result of each sample)
inputData(:,colIn) = [];
% % disp(inputData);
% Each column is a sample, and each row is a feature. So the matrix need to
% do transposition.
inputData = inputData.';
outputData = outputData.';
% % 
disp(size(inputData));
disp(size(outputData));


%% 
% set up BP
net2 = newff(inputData, outputData, [50, 40, 30, 20, 1], {'logsig', 'logsig', 'logsig', 'logsig', 'purelin'}, 'trainscg'); % 50 is hidden layer number

%disp(size(net));

% iteration time
net2.trainParam.epochs = 1000;
% learning rate
net2.trainParam.lr = 0.09;
% goal
net2.trainParam.goal = 0.05;
% max validation check times
net2.trainParam.max_fail = 11;
% train the neural network
net2 = train(net2, inputData, outputData);


%%
% input_test = csvread('sample_train.csv',1,1);
input_test = csvread('test.csv',1,1);
[rowTestIn, colTestIn] = size(input_test);
% disp(width);
% disp(height);
% input_test(:,colTestIn) = [];
input_test = input_test.';
disp(size(input_test));
% % disp(input_test);

RNG0 = [1, 0, rowTestIn, 0];
% orderMatrix = csvread('sample_train.csv',1,0, RNG0);
orderMatrix = csvread('test.csv',1,0, RNG0);
answer = sim(net2, input_test);
answer = answer.';
answer = [orderMatrix, answer];
csvwrite('prediction_submission.csv', answer);
% disp(answer);

