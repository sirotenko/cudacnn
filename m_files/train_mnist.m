%Convolutional neural network for handwriten digits recognition: training
%and simulation.
%(c)Mikhail Sirotenko, 2009.
%This program implements the convolutional neural network for MNIST handwriten 
%digits recognition, created by Yann LeCun. CNN class allows to make your
%own convolutional neural net, defining arbitrary structure and parameters.
%It is assumed that MNIST database is located in './MNIST' directory.
%References:
%1. Y. LeCun, L. Bottou, G. Orr and K. Muller: Efficient BackProp, in Orr, G.
%and Muller K. (Eds), Neural Networks: Tricks of the trade, Springer, 1998
%URL:http://yann.lecun.com/exdb/publis/index.html
%2. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning
%Applied to Document Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998
%URL:http://yann.lecun.com/exdb/publis/index.html
%3. Patrice Y. Simard, Dave Steinkraus, John C. Platt: Best Practices for
%Convolutional Neural Networks Applied to Visual Document Analysis
%URL:http://research.microsoft.com/apps/pubs/?id=68920
%4. Thanks to Mike O'Neill for his great article, wich is summarize and
%generalize all the information in 1-3 for better understandig for
%programming:
%URL: http://www.codeproject.com/KB/library/NeuralNetRecognition.aspx
%5. Also thanks to Jake Bouvrie for his "Notes on Convolutional Neural
%Networks", particulary for the idea to debug the neural network using
%finite differences
%URL: http://web.mit.edu/jvb/www/cv.html

clear;
clc;
%Load the digits into workspace
mnist_train_reader.num_samples = 30000;
mnist_train_reader.current = 1;
mnist_train_reader.data_file = '..\data\MNIST\train-images.idx3-ubyte';
mnist_train_reader.label_file = '..\data\MNIST\train-labels.idx1-ubyte';
mnist_train_reader.buffer_size = 1000;
mnist_train_reader.read = @mnist_datareader;

mnist_test_reader.num_samples = 1000;
mnist_test_reader.current = 1;
mnist_test_reader.data_file = '..\data\MNIST\t10k-images.idx3-ubyte';
mnist_test_reader.label_file = '..\data\MNIST\t10k-labels.idx1-ubyte';
mnist_test_reader.buffer_size = 1000;
mnist_test_reader.read = @mnist_datareader;

%[I,labels,I_test,labels_test] = readMNIST(num_training_samples, '..\data\MNIST\'); 

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
trainer.TrainMethod = 'StochasticLM';
%trainer.TrainMethod = 'StochasticGD';
% HcalcMode = 1 - Mini batch hessian recomputation
% HcalcMode = 2 - Running estimate hessian
trainer.HcalcMode = 1;
trainer.Hrecalc = 300; %Number of iterations to passs for Hessian recalculation
trainer.HrecalcSamplesNum = 30; %Number of samples for Hessian recalculation
trainer.MCRUpdatePeriod = 1000;
trainer.MCRSubsetSize = 300;
trainer.RMSEUpdatePeriod = 30;

%Images preprocessing. Resulting images have 0 mean and 1 standard
%deviation. 
%[inp_train, targ_train] = preproc_mnist_data(I,num_training_samples,labels,0);
%[inp_test, targ_test] = preproc_mnist_data(I_test,100,labels_test,0);
%Actualy training
%cnnet = train(cnnet, trainer, inp_train, targ_train, inp_test, targ_test);
cnnet = train(cnnet, trainer, mnist_train_reader, mnist_test_reader);



