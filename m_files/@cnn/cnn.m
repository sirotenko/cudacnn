function cnet = cnn(cnnet_struct, init_weights)
%CNN convolutional neural network class constructor  
%
%  Syntax
%  
%    cnet =  cnn(cnnet_struct, init_weights)
%    
%  Description
%   Input:
%    cnnet_struct - structure describing the architecture of convolutional
%       neural network
%    init_weights - if true, initialize weights and layers output dimensions
%       by some chosen method
%   Output:
%    cnet - convolutional neural network class object
%
%   
%(c) Sirotenko Mikhail, 2011

if init_weights
    out_width = cnnet_struct.inputWidth;
    out_height = cnnet_struct.inputHeight;
    out_num = cnnet_struct.nInputs;
    
    for k = 1:length(cnnet_struct.layers)
        layer = cnnet_struct.layers{k};
        switch(layer.LayerType)
            case 'clayer'
                cnnet_struct.layers{k}.InpWidth = out_width;
                cnnet_struct.layers{k}.InpHeight = out_height;
                %Fan-in = size of kernel + 1 bias
                %sigma = 1/sqrt(cnnet_struct.layers{k}.KernelWidth*cnnet_struct.layers{k}.KernelHeight*out_num + 1); 
                sigma = 1/sqrt(cnnet_struct.layers{k}.KernelWidth*cnnet_struct.layers{k}.KernelHeight*out_num + 1); 
                %Weights is 4D
                cnnet_struct.layers{k}.Weights = rand(cnnet_struct.layers{k}.KernelWidth, ...
                     cnnet_struct.layers{k}.KernelHeight,out_num, cnnet_struct.layers{k}.NumFMaps) - 0.5;
                cnnet_struct.layers{k}.Weights = cnnet_struct.layers{k}.Weights*sigma/std(cnnet_struct.layers{k}.Weights(:));

                %Biases should be close to zero
                cnnet_struct.layers{k}.Biases = rand(cnnet_struct.layers{k}.NumFMaps,1) - 0.5;
                cnnet_struct.layers{k}.Biases = sigma*cnnet_struct.layers{k}.Biases / std(cnnet_struct.layers{k}.Biases(:));
                %cnnet_struct.layers{k}.Biases = zeros(cnnet_struct.layers{k}.NumFMaps,1);
                %Default connection map - all ones
                if ~isfield(cnnet_struct.layers{k}, 'conn_map')
                    cnnet_struct.layers{k}.conn_map = ones(out_num, cnnet_struct.layers{k}.NumFMaps);
                end
                
                out_width = out_width - (cnnet_struct.layers{k}.KernelWidth - 1);
                out_height = out_height - (cnnet_struct.layers{k}.KernelHeight - 1);
                out_num = cnnet_struct.layers{k}.NumFMaps;
            case 'pooling'
                cnnet_struct.layers{k}.InpWidth = out_width;
                cnnet_struct.layers{k}.InpHeight = out_height;
                cnnet_struct.layers{k}.NumFMaps = out_num;

                out_width = uint32(out_width/cnnet_struct.layers{k}.SXRate);
                out_height = uint32(out_height/cnnet_struct.layers{k}.SYRate);
                out_num = cnnet_struct.layers{k}.NumFMaps;                

            case 'flayer'
                n_inps = out_width*out_height*out_num;
                %sigma = 1/sqrt(n_inps + 1);
                sigma = 1/sqrt(cnnet_struct.layers{k}.numNeurons + 1);
                
                cnnet_struct.layers{k}.Weights = rand(cnnet_struct.layers{k}.numNeurons, n_inps) - 0.5;
                cnnet_struct.layers{k}.Weights = cnnet_struct.layers{k}.Weights*sigma/std(cnnet_struct.layers{k}.Weights(:));
                %Biases should be close to zero
                cnnet_struct.layers{k}.Biases = rand(cnnet_struct.layers{k}.numNeurons,1) - 0.5;
                cnnet_struct.layers{k}.Biases = cnnet_struct.layers{k}.Biases*sigma/std(cnnet_struct.layers{k}.Biases);
                %cnnet_struct.layers{k}.Biases = zeros(cnnet_struct.layers{k}.numNeurons,1);
                
                out_width = 1;
                out_height = 1;
                out_num = cnnet_struct.layers{k}.numNeurons;               

            otherwise
                error('Unknown layer type');
        end
    end
end

cnet = class(cnnet_struct,'cnn');
cnet = validate(cnet,true);
cudacnnMex(cnet,'init');
end
