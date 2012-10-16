function save_cnet08_to_hdf5(cnet_obj, filename)
%Convert to struct
cnet = struct(cnet_obj);
ROOT_LAYERS_GROUP_NAME  = '/Layers';
LAYER_GROUP_NAME  = '/Layer';
hdf5write(filename, [ROOT_LAYERS_GROUP_NAME],'');
hdf5write(filename, [ROOT_LAYERS_GROUP_NAME '/Version'], 1);
% 
% %Workaround: because first layer is always S-layer, check if it does
% %nothing, skip it
if(cnet.SLayer{1}.SRate ~= 1)
    write_slayer(filename, cnet.SLayer{1}, 1);
    offs = 0;
else
    offs = 1;
end

for k = 2:(cnet.numLayers-cnet.numFLayers); %(First layer is dummy, skip it)
    if(rem(k,2)) %Parity check
    %S-layer
        write_slayer(filename, cnet.SLayer{k}, k - offs);
    else
    %C-layer      
        write_clayer(filename, cnet.CLayer{k}, k - offs);
    end
end

for k=(cnet.numLayers-cnet.numFLayers+1):cnet.numLayers
    write_flayer(filename, cnet.FLayer{k}, k - offs);
end

%Write other attributes
%Below -offs stands for ignoring 1st dummy layer
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'nLayers', uint8(cnet.numLayers - offs));
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'nInputs', uint8(cnet.numInputs));
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'inputWidth', uint32(cnet.InputWidth));    
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'inputHeight', uint32(cnet.InputHeight)); 
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'nOutputs', uint32(cnet.numOutputs));    
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'perfFunc', uint32(get_perf_func_enum(cnet.Perf)));    


hdf5writeCustom(filename, ['/TrainParams'], 'mu', single(cnet.mu));        
hdf5writeCustom(filename, ['/TrainParams'], 'muDec', single(cnet.mu_dec));        
hdf5writeCustom(filename, ['/TrainParams'], 'muInc', single(cnet.mu_inc));        
hdf5writeCustom(filename, ['/TrainParams'], 'muMax', single(cnet.mu_max));
hdf5writeCustom(filename, ['/TrainParams'], 'epochs', uint32(cnet.epochs));        
hdf5writeCustom(filename, ['/TrainParams'], 'goal', single(cnet.goal));        
hdf5writeCustom(filename, ['/TrainParams'], 'teta', single(cnet.teta));        
hdf5writeCustom(filename, ['/TrainParams'], 'tetaDec', single(cnet.teta_dec)); 

end

% function hdf5writeAttribute(filename, location, name, data)
%     attr = data;
%     attr_details.Name = name;
%     attr_details.AttachedTo = location;
%     attr_details.AttachType = 'group';
% 
%     hdf5write(filename, attr_details, attr);
% end

function write_slayer(filename, slayer, lnum)
    weights_mat = convert_cell_to_3d(slayer.WS);
    biases_mat = convert_cell_to_3d(slayer.BS);
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Weights', single(weights_mat));%,'V71Dimensions',true);        
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Biases', single(biases_mat));%,'V71Dimensions',true);        
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerType', uint32(1));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerNumber', uint32(lnum));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'PoolingType', uint32(0));    %Subsampling
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'TransferFunc', ...
        uint32(get_trasfer_func_enum(slayer.TransfFunc)));    
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'NumFMaps', uint32(slayer.numFMaps));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'OutFMapWidth', uint32(slayer.FMapWidth));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'OutFMapHeight', uint32(slayer.FMapHeight));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'SXRate', uint32(slayer.SRate));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'SYRate', uint32(slayer.SRate));
end

function write_clayer(filename, clayer, lnum)
    weights_mat = convert_cell_to_3d(clayer.WC);
    biases_mat =  convert_cell_to_3d(clayer.BC);
    con_map = convert_cell_to_3d(clayer.ConMap);
    %hdf5write(filename, ['/Layers/Layer' num2str(lnum) '/Weights'], single(weights_mat));%,'V71Dimensions',true);        
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Weights', single(weights_mat));%,'V71Dimensions',true);        
    %hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum) '/Weights'], single(weights_mat));%,'V71Dimensions',true);        
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Biases', single(biases_mat));%, 'V71Dimensions',true);  
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'ConnMap', uint32(con_map));%,'V71Dimensions',true);  
    
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerType', uint32(0));
    
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerType', uint32(0));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerNumber', uint32(lnum));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'NumFMaps', uint32(clayer.numFMaps));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'OutFMapWidth', uint32(clayer.FMapWidth));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'OutFMapHeight', uint32(clayer.FMapHeight));

    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'TransferFunc', ...
        uint32(0));    %Linear

end

function write_flayer(filename, flayer, lnum)
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Weights', single(flip_dim(flayer.W)));%);%,'V71Dimensions',true);        
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Biases', single(flip_dim(flayer.B)));%);%,'V71Dimensions',true);        
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerType', uint32(2));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerNumber', uint32(lnum));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'NumNeurons', uint32(flayer.numNeurons));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'TransferFunc', ...
        uint32(get_trasfer_func_enum(flayer.TransfFunc)));    %Subsampling

end

%Exchange dimensions for row-wise format
function out = flip_dim(x)
    x = x';
    sz = size(x);
    out = reshape(x, sz(2), sz(1));
end

%Function transposes each martix in cell array and converts to 
%to multidimensional
function out = convert_cell_to_3d(inp)
    if(iscell(inp))
        %Represent cell array as 3d array
        sz = size(inp{1});
        inp = cellfun(@(x) x', inp,  'UniformOutput', false);
        %inp = cellfun(@flip_dim, inp,  'UniformOutput', false);
        
        out = reshape(cell2mat(inp),length(inp),sz(1),sz(2));
        %out = reshape(cell2mat(inp),sz(1),sz(2),length(inp));
    else
        %out = inp';
        out = flip_dim(inp);
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

function out = get_perf_func_enum(in)
    switch in
        case 'mse' 
            out = 0;
        case 'lse' 
            out = 1;
        otherwise
            out = 0;
    end
end