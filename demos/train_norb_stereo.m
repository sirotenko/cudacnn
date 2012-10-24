%Train on monocular NORB dataset on recognition/detection task
% 6 outputs: 5 classes and 1 for background.


clear;
clc;
%Load the digits into workspace
norb_train_reader.num_samples = 291600;
norb_train_reader.current = 1;
norb_train_reader.data_files = {'..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-01-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-02-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-03-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-04-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-05-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-06-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-07-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-08-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-09-dat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-10-dat.mat',...                                 
                                 };
norb_train_reader.label_files = {'..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-01-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-02-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-03-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-04-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-05-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-06-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-07-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-08-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-09-cat.mat',...
                                 '..\data\NORB\norb-5x46789x9x18x6x2x108x108-training-10-cat.mat',...
                                };
norb_train_reader.buffer_size = 1000;
norb_train_reader.stereo = true;
norb_train_reader.read = @norb_datareader;

norb_test_reader.num_samples = 1000;
norb_test_reader.current = 1;
norb_test_reader.data_files = {'..\data\NORB\norb-5x01235x9x18x6x2x108x108-testing-01-dat.mat',...
                                 '..\data\NORB\norb-5x01235x9x18x6x2x108x108-testing-02-dat.mat',...
                                 };
norb_test_reader.label_files = {'..\data\NORB\norb-5x01235x9x18x6x2x108x108-testing-01-cat.mat',...
                                 '..\data\NORB\norb-5x01235x9x18x6x2x108x108-testing-02-cat.mat',...
                                };
norb_test_reader.buffer_size = 1000;
norb_test_reader.stereo = true;
norb_test_reader.read = @norb_datareader;
%%
%Define the network architecture
%Total number of layers
cnet_struct.nlayers = 7;
%Number of input images (simultaneously processed). 
cnet_struct.nInputs = 2;
%Image width
cnet_struct.inputWidth = 108;    
%Image height
cnet_struct.inputHeight = 108; 


cnet_struct.layers{1}.NumFMaps = 8;
cnet_struct.layers{1}.KernelWidth = 5;
cnet_struct.layers{1}.KernelHeight = 5;

cnet_struct.layers{1}.TransferFunc = 'tansig_mod';
cnet_struct.layers{1}.LayerType = 'clayer';
cnet_struct.layers{1}.conn_map = ...
[1 1 0 0 1 1 1 1;
 0 0 1 1 1 1 1 1;
];


cnet_struct.layers{2}.SXRate = 4;
cnet_struct.layers{2}.SYRate = 4;
cnet_struct.layers{2}.PoolingType = 'max';
cnet_struct.layers{2}.LayerType = 'pooling';


cnet_struct.layers{3}.NumFMaps = 20;
cnet_struct.layers{3}.KernelWidth = 6;
cnet_struct.layers{3}.KernelHeight = 6;
cnet_struct.layers{3}.TransferFunc = 'tansig_mod';
cnet_struct.layers{3}.LayerType = 'clayer';

cnet_struct.layers{3}.conn_map = ...
[1 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1;
 1 1 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1;
 1 1 1 0 0 0 1 1 1 0 0 1 0 1 1 1 1 1 1 1;
 0 1 1 1 0 0 1 1 1 1 0 0 1 0 1 1 1 1 1 1;
 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1 1 1 1; 
 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1 1 1 1 1 1; 
];

cnet_struct.layers{4}.SXRate = 3;
cnet_struct.layers{4}.SYRate = 3;
cnet_struct.layers{4}.PoolingType = 'max';
cnet_struct.layers{4}.LayerType = 'pooling';

cnet_struct.layers{5}.NumFMaps = 100;
cnet_struct.layers{5}.KernelWidth = 6;
cnet_struct.layers{5}.KernelHeight = 6;
cnet_struct.layers{5}.TransferFunc = 'tansig_mod';

cnet_struct.layers{5}.LayerType = 'clayer';


cnet_struct.layers{6}.numNeurons = 20;
cnet_struct.layers{6}.TransferFunc = 'tansig_mod';
cnet_struct.layers{6}.LayerType = 'flayer';

cnet_struct.layers{7}.numNeurons = 6;
cnet_struct.layers{7}.TransferFunc = 'tansig_mod';
cnet_struct.layers{7}.LayerType = 'flayer';

%Create convolutional neural network class object, initialize weights and
%biases
cnnet = cnn(cnet_struct,true);
%%
%Initialize trainer
%Number of epochs
%trainer.epochs = 10;
trainer.epochs = 5;
%Mu coefficient for stochastic Levenberg-Markvardt
trainer.mu = 0.0000001;
%Training coefficient
%trainer.theta =  [50 30 20 10 5 5 5 1 1 1]/100000;
trainer.theta =  [20 10 5 1 1]/100000;
trainer.TrainMethod = 'StochasticLM';
%trainer.TrainMethod = 'StochasticGD';
%HcalcMode = 1 - Mini batch hessian recomputation
% HcalcMode = 2 - Running estimate hessian
trainer.HcalcMode = 1;
trainer.Hrecalc = 800; %Number of iterations to passs for Hessian recalculation
trainer.HrecalcSamplesNum = 50; %Number of samples for Hessian recalculation
trainer.MCRUpdatePeriod = 1000;
trainer.MCRSubsetSize = 300;
trainer.RMSEUpdatePeriod = 30;

%Images preprocessing. Resulting images have 0 mean and 1 standard
%deviation. 
%Actualy training
cnnet = train(cnnet, trainer, norb_train_reader, norb_test_reader);

