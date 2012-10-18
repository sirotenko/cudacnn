function save(cnnet, filename)
%TODO: change LayerType to string
%SAVE save convolutional neural network in HDF5 file  
%
%  Syntax
%  
%    save(cnnet)
%
%  Description
%   Input:
%    cnnet_struct - convolutional neural network class object
%
%
%(c) Sirotenko Mikhail, 2011

ROOT_LAYERS_GROUP_NAME  = '/Layers';
LAYER_GROUP_NAME  = '/Layer';
hdf5write(filename, [ROOT_LAYERS_GROUP_NAME],'');
hdf5write(filename, [ROOT_LAYERS_GROUP_NAME '/Version'], 1);

hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'nLayers', uint8(cnnet.nlayers));
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'nInputs', uint8(cnnet.nInputs));
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'inputWidth', uint32(cnnet.inputWidth));    
hdf5writeAttribute(filename, ROOT_LAYERS_GROUP_NAME, 'inputHeight', uint32(cnnet.inputHeight)); 


for k = 1:length(cnnet.layers)
    layer = cnnet.layers{k};
    switch(layer.LayerType)
        case 'clayer'
            write_clayer(filename, cnnet.layers{k}, k);
        case 'pooling'
            write_player(filename, cnnet.layers{k}, k);            
        case 'flayer'
            write_flayer(filename, cnnet.layers{k}, k );
        otherwise
            error('Unknown layer type');
    end
end
end

function write_player(filename, player, lnum)
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerType', uint32(2));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerNumber', uint32(lnum));
    pool_type = uint32(get_pooling_enum(player.PoolingType));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'PoolingType', pool_type);    %Subsampling
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'NumFMaps', uint32(player.NumFMaps));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'InpWidth', uint32(player.InpWidth));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'InpHeight', uint32(player.InpHeight));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'SXRate', uint32(player.SXRate));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'SYRate', uint32(player.SYRate));
end

function write_clayer(filename, clayer, lnum)
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerType', uint32(1));

    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Weights', clayer.Weights);
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Biases', clayer.Biases);
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'ConnMap', clayer.conn_map);
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'TransferFunc', ...
        uint32(get_trasfer_func_enum(clayer.TransferFunc)));    
    
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerNumber', uint32(lnum));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'NumFMaps', uint32(clayer.NumFMaps));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'InpWidth', uint32(clayer.InpWidth));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'InpHeight', uint32(clayer.InpHeight));
end

function write_flayer(filename, flayer, lnum)
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Weights', flayer.Weights);
    hdf5writeCustom(filename, ['/Layers/Layer' num2str(lnum)], 'Biases', flayer.Biases);
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerType', uint32(3));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'LayerNumber', uint32(lnum));
    hdf5writeAttribute(filename, ['/Layers/Layer' num2str(lnum)], 'TransferFunc', ...
        uint32(get_trasfer_func_enum(flayer.TransferFunc)));   
end

function out = get_trasfer_func_enum(in)
    switch in
        case 'purelin' 
            out = 1;
        case 'tansig_mod' 
            out = 2;
        case 'tansig' 
            out =  3;
        otherwise
            out = 0;
    end
end

function out = get_pooling_enum(in)
    switch in
        case 'average' 
            out = 1;
        case 'max' 
            out = 2;
        otherwise
            out = 0;
    end
end

function hdf5writeAttribute(filename, path, name, data)

acpl = H5P.create('H5P_ATTRIBUTE_CREATE');
aapl = 'H5P_DEFAULT';
switch(class(data))
    case 'uint32'
        type_id = H5T.copy('H5T_NATIVE_UINT');
    case 'double'
        type_id = H5T.copy('H5T_NATIVE_DOUBLE');
    case 'float'
        type_id = H5T.copy('H5T_NATIVE_FLOAT');
    case 'uint8'
        type_id = H5T.copy('H5T_NATIVE_UCHAR');
    otherwise
        error(['Unsupported data type: ' class(data)]);

end

fid = H5F.open(filename,'H5F_ACC_RDWR','H5P_DEFAULT');
plist = 'H5P_DEFAULT';
%H5 Lib is exception based
try
    gid = H5G.open(fid, path);
catch exception
    if (strcmp(exception.identifier, ...
   'MATLAB:hdf5lib2:H5Gopen1:failure'))
        gid = H5G.create(fid, path, plist,plist,plist);
    else
        rethrow(exception);
    end
end

try
    attr_id = H5A.open(gid,name,'H5P_DEFAULT');
catch exception
    if (strcmp(exception.identifier, ...
   'MATLAB:hdf5lib2:H5Aopen:failure'))
        space_id = H5S.create('H5S_SCALAR');
        attr_id = H5A.create(gid,name,type_id,space_id,acpl,aapl);
    else
        rethrow(exception);
    end    
end

H5A.write(attr_id, type_id,data);

H5A.close(attr_id);
H5G.close(gid);
H5F.close(fid);
end

function hdf5writeCustom(filename, path, name, mat)
% if(~isa(mat,'single'))
%     error('Input should be single');
% end
switch(class(mat))
    case 'double'
        type_id = H5T.copy('H5T_NATIVE_DOUBLE');
    case 'single'
        type_id = H5T.copy('H5T_NATIVE_FLOAT');
    case 'uint32'
        type_id = H5T.copy('H5T_NATIVE_UINT32');
    case 'int32'
        type_id = H5T.copy('H5T_NATIVE_INT32');

    otherwise
        error('Unsupported type');
end
plist = 'H5P_DEFAULT';
fid = H5F.open(filename,'H5F_ACC_RDWR','H5P_DEFAULT');

%H5 Lib is exception based
try
    gid = H5G.open(fid, path);
catch exception
    if (strcmp(exception.identifier, ...
   'MATLAB:hdf5lib2:H5Gopen1:failure'))
        gid = H5G.create(fid, path, plist,plist,plist);
    else
        rethrow(exception);
    end
end

if(isscalar(mat))
    space_id = H5S.create('H5S_SCALAR');
else
    dims = size(mat);
    h5_dims = fliplr(dims);
    h5_maxdims = h5_dims;
    space_id = H5S.create_simple(ndims(mat),h5_dims,h5_maxdims);
end
dset_id = H5D.create(gid,name,type_id,space_id,plist,plist,plist);
%dset_id = H5D.open(fid,path);
H5D.write(dset_id,'H5ML_DEFAULT','H5S_ALL','H5S_ALL',plist,mat);
H5S.close(space_id);
H5T.close(type_id);
H5D.close(dset_id);
H5G.close(gid);
H5F.close(fid);
end