function cnet_20 = ConvertTo20Format(cnet, varargin)
%ConvertTo20Format convert convolutional neural network ver <0.9 to >=2.0 
%
%  Syntax
%  
%    cnet_20 = ConvertTo20Format(cnet, varargin)
%    
%  Description
%   Input:
%    cnet - convolutional network of old format
%    varargin - if non empty than double precision is used and single by
%    default
%   Output:
%    cnet - convolutional neural network version 2.0
%
%   Convert CNN to new format. All matrices are transposed and converted to
%   some single or double precision.
%
%(c) Sirotenko Mikhail, 2011

if(~isempty(varargin))
    precisionFn = @double;
else
    precisionFn = @single;
end

cnet_20.nlayers = uint8(cnet.numFLayers + cnet.numCLayers + cnet.numSLayers);
cnet_20.nInputs = uint8(cnet.numInputs);
cnet_20.inputWidth = uint32(cnet.InputWidth);    
cnet_20.inputHeight = uint32(cnet.InputHeight); 

for k=1:(cnet.numLayers-cnet.numFLayers) 
    if(rem(k,2)) %Parity check
        %S-layer
        %cnet_20.layers{k}.W = cellfun(@(x) transp(precisionFn(x)),cnet.SLayer{k}.WS,'UniformOutput',0);
        %cnet_20.layers{k}.B = cellfun(@(x) transp(precisionFn(x)),cnet.SLayer{k}.BS,'UniformOutput',0);            
        if(~isempty(cnet.SLayer{k}.WS))
            warning('Weights and biases in pooling layers are omitted in CNN ver 1.0. Weights in SLayer ignored.');
        end
        if(~isempty(cnet.SLayer{k}.BS))
            warning('Weights and biases in pooling layers are omitted in CNN ver 1.0. Biases in SLayer ignored.');
        end
        cnet_20.layers{k}.NumFMaps = uint32(cnet.SLayer{k}.numFMaps);
        cnet_20.layers{k}.OutFMapWidth = uint32(cnet.SLayer{k}.FMapWidth);
        cnet_20.layers{k}.OutFMapHeight = uint32(cnet.SLayer{k}.FMapHeight);

        cnet_20.layers{k}.TransferFunc = cnet.SLayer{k}.TransfFunc;
        cnet_20.layers{k}.SXRate = uint32(cnet.SLayer{k}.SRate);
        cnet_20.layers{k}.SYRate = uint32(cnet.SLayer{k}.SRate);
        cnet_20.layers{k}.LayerType = 'subsampling';
    else
    %C-layer      
        cnet_20.layers{k}.NumFMaps = uint32(cnet.CLayer{k}.numFMaps);
        cnet_20.layers{k}.OutFMapWidth = uint32(cnet.CLayer{k}.FMapWidth);
        cnet_20.layers{k}.OutFMapHeight = uint32(cnet.CLayer{k}.FMapHeight);

%         cnet_20.layers{k}.W = cell2mat(cellfun(@(x) transp(precisionFn(x)),cnet.CLayer{k}.WC,'UniformOutput',0);
%         cnet_20.layers{k}.B = cellfun(@(x) transp(precisionFn(x)),cnet.CLayer{k}.BC,'UniformOutput',0);
        for i = 1:length(cnet.CLayer{k}.WC)
            cnet_20.layers{k}.Weights(:,:,i) = precisionFn(cnet.CLayer{k}.WC{i})';
            cnet_20.layers{k}.Biases(:,:,i) = precisionFn(cnet.CLayer{k}.BC{i})';
        end
        cnet_20.layers{k}.conn_map = int32(cnet.CLayer{k}.ConMap)';
        cnet_20.layers{k}.LayerType = 'clayer';
    end
end

for k=(cnet.numLayers-cnet.numFLayers+1):cnet.numLayers
    cnet_20.layers{k}.Weights = precisionFn(cnet.FLayer{k}.W)';
    cnet_20.layers{k}.Biases = precisionFn(cnet.FLayer{k}.B)';                
    cnet_20.layers{k}.LayerType = 'flayer';
    cnet_20.layers{k}.TransferFunc = cnet.FLayer{k}.TransfFunc;
end
end

function out = get_trasfer_func_enum(in)
    switch in
        case 'purelin' 
            out = 0;
        case 'tansig_mod' 
            out = 1;
        case 'tansig' 
            out =  2;
        otherwise
            out = 0;
    end
end
