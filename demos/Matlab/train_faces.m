%Demo script for training convolutional neural network for face detection
%(c)Mikhail Sirotenko, 2012.

clear classes;
clear;
clc;

addpath('../../m_files');

%%
%Load the faces into workspace
load('../../data/FacesMIT/faces_train.mat');
load('../../data/FacesMIT/non_faces_train.mat');
load('../../data/FacesMIT/faces_test.mat');
load('../../data/FacesMIT/non_faces_test.mat');

mitface_train_reader.num_faces = length(faces_train);
mitface_train_reader.num_nonfaces = length(nonfaces_train);
mitface_train_reader.num_samples = length(faces_train) + length(nonfaces_train);
mitface_train_reader.current = 1;
mitface_train_reader.faces = faces_train;
mitface_train_reader.nonfaces = nonfaces_train;
mitface_train_reader.read = @mitfaces_datareader;

mitface_test_reader.num_samples = length(faces_test) + length(nonfaces_test);
mitface_test_reader.current = 1;
mitface_train_reader.faces = faces_test;
mitface_train_reader.nonfaces = nonfaces_test;

mitface_test_reader.read = @mitfaces_datareader;

%%

%Define the structure according to [2]

%Total number of layers
cnet_struct.nlayers = 7;
%Number of input images (simultaneously processed). 
cnet_struct.nInputs = 1;
%Image width
cnet_struct.inputWidth = 32;    
%Image height
cnet_struct.inputHeight = 32; 


%Now define the network parameters

%First layer - 6 convolution kernels with 5x5 size 
cnet_struct.layers{1}.NumFMaps = 6;
cnet_struct.layers{1}.KernelWidth = 5;
cnet_struct.layers{1}.KernelHeight = 5;
%Define transfer function
cnet_struct.layers{1}.TransferFunc = 'tansig_mod';
cnet_struct.layers{1}.LayerType = 'clayer';

%Weights 150
%Biases 6

%Second layer
%Subsampling rate
cnet_struct.layers{2}.SXRate = 2;
cnet_struct.layers{2}.SYRate = 2;
cnet_struct.layers{2}.PoolingType = 'max';
cnet_struct.layers{2}.LayerType = 'pooling';


%Weights 6
%Biases 6

%Third layer - 16 kernels with 5x5 size 
cnet_struct.layers{3}.NumFMaps = 16;
cnet_struct.layers{3}.KernelWidth = 5;
cnet_struct.layers{3}.KernelHeight = 5;
cnet_struct.layers{3}.TransferFunc = 'tansig_mod';
cnet_struct.layers{3}.LayerType = 'clayer';
%According to [2] the generalisation is better if there's unsimmetry in
%layers connections. Yann LeCun uses this kind of connection map:
cnet_struct.layers{3}.conn_map = ...
[1 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1;
 1 1 0 0 0 1 1 1 0 0 1 1 1 1 0 1;
 1 1 1 0 0 0 1 1 1 0 0 1 0 1 1 1;
 0 1 1 1 0 0 1 1 1 1 0 0 1 0 1 1;
 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1; 
 0 0 0 1 1 1 0 0 1 1 1 1 0 1 1 1; 
];

%Weights 150
%Biases 6

%Fourth layer
%Subsampling rate
cnet_struct.layers{4}.SXRate = 2;
cnet_struct.layers{4}.SYRate = 2;
cnet_struct.layers{4}.PoolingType = 'max';
cnet_struct.layers{4}.LayerType = 'pooling';


%Weights 6
%Biases 6

%Fifth layer - outputs 120 feature maps 1x1 size
cnet_struct.layers{5}.NumFMaps = 120;
cnet_struct.layers{5}.KernelWidth = 5;
cnet_struct.layers{5}.KernelHeight = 5;
cnet_struct.layers{5}.TransferFunc = 'tansig_mod';

cnet_struct.layers{5}.LayerType = 'clayer';

%Seventh layer - fully connected, 84 neurons
cnet_struct.layers{6}.numNeurons = 84;
cnet_struct.layers{6}.TransferFunc = 'tansig_mod';
cnet_struct.layers{6}.LayerType = 'flayer';
%Eight layer - fully connected, 10 output neurons
cnet_struct.layers{7}.numNeurons = 10;
cnet_struct.layers{7}.TransferFunc = 'tansig_mod';
cnet_struct.layers{7}.LayerType = 'flayer';
%Create convolutional neural network class object, initialize weights and
%biases
cnnet = cnn(cnet_struct,true);

%%
%Initialize trainer
%Number of epochs
trainer.epochs = 10;
%Mu coefficient for stochastic Levenberg-Markvardt
trainer.mu = 0.0000001;
%Training coefficient
trainer.theta =  [50 30 20 10 5 5 5 1 1 1]/10000;
%trainer.TrainMethod = 'StochasticLM';
trainer.TrainMethod = 'StochasticGD';
% HcalcMode = 1 - Mini batch hessian recomputation
% HcalcMode = 2 - Running estimate hessian
trainer.HcalcMode = 0;
trainer.Hrecalc = 1000; %Number of iterations to passs for Hessian recalculation
trainer.HrecalcSamplesNum = 30; %Number of samples for Hessian recalculation
trainer.MCRUpdatePeriod = 1000;
trainer.MCRSubsetSize = 300;
trainer.RMSEUpdatePeriod = 30;

%Images preprocessing. Resulting images have 0 mean and 1 standard
%deviation. 
%Actualy training
cnnet = train(cnnet, trainer, mitface_train_reader, mitface_test_reader);



