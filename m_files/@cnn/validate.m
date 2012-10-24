% Check if it contains all necessary fields and there're no type missmatch
function cnnet_out = validate(cnnet, convert_types)
%validate - check the presence of all necessary fields and their types. If 
%specified, convert types.
%
%  Syntax
%  
%    cnnet_out = validate(cnnet, convert_types)
%    
%  Description
%   Input:
%    cnet - Convolutional neural network struct
%    convert_types - if true, then all incompatible types will be converted
%           otherwise it will be treated as error
%   Output:
%    cnnet_out - cnn object with correct type fields 
%   
% Function will check the correctness of neural network stucture structure
%(c) Sirotenko Mikhail, 2011

%valid_layer_types = {'clayer', 'flayer', 'subsampling'};
valid_transfer_functions = {'tansig', 'tansig_mod', 'purelin'};

if(cnnet.nlayers ~= length(cnnet.layers))
    error('Actual number of layers and "nlayers" field missmatch');
end

input.w = cnnet.inputWidth;
input.h = cnnet.inputHeight;
input.m = cnnet.nInputs;
for i = 1:int32(cnnet.nlayers)
    switch(cnnet.layers{i}.LayerType)
        case 'clayer'
            cnnet.layers{i} = validate_clayer_types(cnnet.layers{i}, i, convert_types);
            if ~ismember(cnnet.layers{i}.TransferFunc,valid_transfer_functions)
                error([cnnet.layers{i}.TransferFunc ' is not a valid transfer function.']);
            end
 
            [kw,kh,km,kn] = size(cnnet.layers{i}.Weights);
            if(cnnet.layers{i}.NumFMaps ~= kn)
                error(['NumFMaps field dont match number of outputs for clayer # ' ...
                    num2str(i) '. NumFMaps = ' num2str(cnnet.layers{i}.NumFMaps) ...
                    ', # of outputs is ' num2str(kn)]);
            end
            if(input.m ~= km)
                error(['Number of kernels not correspond to number of inputs for clayer # ' ...
                    num2str(i) '. Number of input kernels = ' num2str(cnnet.layers{i}.NumFMaps) ...
                    ', # of inputs is ' num2str(kn)]);
            end
            
            true_inp_width = input.w;
            true_inp_height = input.h;
            
            if(cnnet.layers{i}.InpWidth ~= true_inp_width)
                error(['InpWidth dont match with real output with for clayer # ' ...
                    num2str(i) '. InpWidth = ' num2str(cnnet.layers{i}.InpWidth) ...
                    ', actual is ' num2str(true_inp_width)]);
            end

            if(cnnet.layers{i}.InpHeight ~= true_inp_height)
                error(['InpHeight dont match with real output with for clayer # ' ...
                    num2str(i) '. InpHeight = ' num2str(cnnet.layers{i}.InpHeight) ...
                    ', actual is ' num2str(true_inp_height)]);
            end
            if(numel(cnnet.layers{i}.Biases) ~= kn)
                error(['Number of biases should be equal to number of output kernels in clayer # ' ...
                    num2str(i) '. # of Biases = ' num2str(length(cnnet.layers{i}.Biases)) ...
                    '# of output kernels' num2str(kn)]);
            end
%             if(ndims(cnnet.layers{i}.Biases) ~= 1)
%                 error(['Biases should be a vector ']);                 
%             end            
            if(size(cnnet.layers{i}.conn_map,1) ~= input.m)
                error(['conn_map number of rows should be equal to number of inputs in clayer # ' ...
                    num2str(i) '.']);
            end
            if(size(cnnet.layers{i}.conn_map,1) ~= km)
                error(['conn_map number of rows should be equal to number of input kernels in clayer # ' ...
                    num2str(i) '.']);
            end
            
            if(size(cnnet.layers{i}.conn_map,2) ~= kn)
                error(['conn_map number of columnts should be equal to number of output kernels in clayer # ' ...
                    num2str(i) '.']);
            end
            %Set inputs for the  next layer
            input.w = input.w - kw + 1;
            input.h = input.h - kh + 1;
            input.m = kn;
            
        case 'flayer'
            cnnet.layers{i} = validate_flayer_types(cnnet.layers{i}, i, convert_types);
            %Flatten input
            num_inps = input.w * input.h * input.m;
            
            if(size(cnnet.layers{i}.Weights,2) ~= num_inps)
                error(['Number of inputs should be equal to the number of'...
                    ' columns of the weight matrix in flayer # ' ...
                    num2str(i) '. # of inputs = ' num2str(num_inps) ...
                    '. # of columns ' num2str(size(cnnet.layers{i}.Weights,2))]);
            end
            num_outs = size(cnnet.layers{i}.Weights,1);
            if(numel(cnnet.layers{i}.Biases) ~= num_outs)
                error(['Number of biases should be equal to the number of'...
                    ' outputs in flayer # ' ...
                    num2str(i) '. # of biases = ' num2str(numel(cnnet.layers{i}.Biases)) ...
                    '. # of outputs ' num2str(num_outs)]);
            end
            if ~ismember(cnnet.layers{i}.TransferFunc,valid_transfer_functions)
                error([cnnet.layers{i}.TransferFunc ' is not a valid transfer function.']);
            end
            input.w = 1;
            input.h = 1;
            input.m = num_outs;            
        case 'pooling'
            cnnet.layers{i} = validate_player_types(cnnet.layers{i}, i, convert_types);
            if(cnnet.layers{i}.NumFMaps ~= input.m)
                error(['NumFMaps field dont match number of inputs for player # ' ...
                    num2str(i) '. NumFMaps = ' num2str(cnnet.layers{i}.NumFMaps) ...
                    ', # of inputs is ' num2str(input.m)]);
            end
            if(cnnet.layers{i}.SXRate < 1 || cnnet.layers{i}.SYRate < 1)
                error(['Subsampling rate should be > 0. Error in layer #' num2str(i)]);
            end
            
            true_inp_width = uint32(input.w);
            true_inp_height = uint32(input.h);
            
            if(cnnet.layers{i}.InpWidth ~= true_inp_width)
                error(['InpWidth dont match with real output with for layer # ' ...
                    num2str(i) '. InpWidth = ' num2str(cnnet.layers{i}.InpWidth) ...
                    ', actual is ' num2str(true_inp_width)]);
            end

            if(cnnet.layers{i}.InpHeight ~= true_inp_height)
                error(['InpHeight dont match with real output with for layer # ' ...
                    num2str(i) '. InpHeight = ' num2str(cnnet.layers{i}.InpHeight) ...
                    ', actual is ' num2str(true_inp_height)]);
            end
            %Set inputs for the  next layer
            input.w = ceil(uint32(input.w)/cnnet.layers{i}.SXRate);
            input.h = ceil(uint32(input.h)/cnnet.layers{i}.SYRate);
            input.m = cnnet.layers{i}.NumFMaps;
        otherwise
            if(~ischar(cnnet.layers{i}.LayerType))
                error(['LayerType should be string. Error in layer ' num2str(i)]);
            end
            error(['Unknown type of layer: ' cnnet.layers{i}.LayerType '. Error in layer ' num2str(i)]);
    end
end
cnnet_out = cnnet;
end

function [var_out, res] = validate_type(var, cls, convert)
if convert
    var_out = cast(var, cls);
    res = true;
else
    if ~strcmp(class(var), cls)
        res = false;
        var_out = NaN;
    else
        var_out = var;
        res = true;
    end
end
end

function out_clayer = validate_clayer_types(clayer, num, convert_types)
%This templates contain correct variables of correct types
clayer_tmpl.NumFMaps = uint32([]);
clayer_tmpl.InpWidth = uint32([]);
clayer_tmpl.InpHeight = uint32([]);
clayer_tmpl.Weights = single([]);
clayer_tmpl.Biases = single([]);
clayer_tmpl.conn_map = int32([]);
clayer_tmpl.TransferFunc = char([]);
names = fieldnames(clayer_tmpl);
for i = 1:length(names)
    nm = names{i};
    if ~isfield(clayer, nm)
        error(['Field ' nm ' is missing in clayer # ' num2str(num)]);
    end
    [new_fld, res] = validate_type(getfield(clayer,nm), class(getfield(clayer_tmpl,nm)), convert_types);
    if ~res
        error(['CLayer field ' nm ' type missmatch. Should be ' class(getfield(clayer_tmpl,nm))...
            '. Actual ' class(getfield(clayer,nm))]);
    else
        clayer = setfield(clayer,nm,new_fld);
    end
end
out_clayer = clayer;
end

function out_flayer = validate_flayer_types(flayer, num, convert_types)
%This templates contain correct variables of correct types
flayer_tmpl.Weights = single([]);
flayer_tmpl.Biases = single([]);
flayer_tmpl.TransferFunc = char([]);
names = fieldnames(flayer_tmpl);
for i = 1:length(names)
    nm = names{i};
    if ~isfield(flayer, nm)
        error(['Field ' nm ' is missing in flayer # ' num2str(num)]);
    end
    [new_fld, res] = validate_type(getfield(flayer,nm), class(getfield(flayer_tmpl,nm)), convert_types);
    if ~res
        error(['Flayer field ' nm ' type missmatch. Should be ' class(getfield(flayer_tmpl,nm))...
            '. Actual ' class(getfield(flayer,nm))]);
    else
        flayer = setfield(flayer,nm,new_fld);
    end
end
out_flayer = flayer;
end

function out_slayer = validate_player_types(player, num, convert_types)
%This templates contain correct variables of correct types
player_tmpl.NumFMaps = uint32([]);
player_tmpl.InpWidth = uint32([]);
player_tmpl.InpHeight = uint32([]);
player_tmpl.SXRate = uint32([]);
player_tmpl.SYRate = uint32([]);
player_tmpl.PoolingType = char([]);
names = fieldnames(player_tmpl);
for i = 1:length(names)
    nm = names{i};
    if ~isfield(player, nm)
        error(['Field ' nm ' is missing in player # ' num2str(num)]);
    end
    [new_fld, res] = validate_type(getfield(player,nm), class(getfield(player_tmpl,nm)), convert_types);
    if ~res
        error(['Player field ' nm ' type missmatch. Should be ' class(getfield(player_tmpl,nm))...
            '. Actual ' class(getfield(player,nm))]);
    else
        player = setfield(player,nm,new_fld);
    end
end
out_slayer = player;
end