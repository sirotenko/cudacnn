% Check if it contains all necessary fields and there're no type missmatch
function trainer_out = validate_trainer(trainer, convert_types)
%validate_trainer - check the presence of all necessary fields and their types. If 
%specified, convert types.
%
%  Syntax
%  
%    trainer_out = validate_trainer(trainer, convert_types)
%    
%  Description
%   Input:
%    trainer - structure with training related settings
%    convert_types - if true, then all incompatible types will be converted
%           otherwise it will be treated as error
%   Output:
%    trainer_out - trainer struct with correct data types
%   
%(c) Sirotenko Mikhail, 2011

[trainer.epochs, res] = validate_type(trainer.epochs, 'uint32',convert_types);
if ~res
    error(['trainer.epochs type missmatch. Should be uint32' ...
        '. Actual ' class(trainer.epochs)]);    
end
[trainer.theta, res] = validate_type(trainer.theta, 'single',convert_types);
if ~res
    error(['trainer.theta type missmatch. Should be single' ...
        '. Actual ' class(trainer.theta)]);    
end

switch(trainer.TrainMethod)
    case 'StochasticLM'
        [trainer.mu, res] = validate_type(trainer.mu, 'single',convert_types);
        if ~res
            error(['trainer.mu type missmatch. Should be single' ...
                '. Actual ' class(trainer.mu)]);
        end        
    case 'StochasticGD'
    otherwise
        error('Unknown train method');
end

trainer_out = trainer;
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
clayer_tmpl.OutFMapWidth = uint32([]);
clayer_tmpl.OutFMapHeight = uint32([]);
clayer_tmpl.Weights = single([]);
clayer_tmpl.Biases = single([]);
clayer_tmpl.conn_map = int32([]);
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

function out_slayer = validate_slayer_types(slayer, num, convert_types)
%This templates contain correct variables of correct types
slayer_tmpl.NumFMaps = uint32([]);
slayer_tmpl.OutFMapWidth = uint32([]);
slayer_tmpl.OutFMapHeight = uint32([]);
slayer_tmpl.SXRate = uint32([]);
slayer_tmpl.SYRate = uint32([]);
slayer_tmpl.TransferFunc = char([]);
names = fieldnames(slayer_tmpl);
for i = 1:length(names)
    nm = names{i};
    if ~isfield(slayer, nm)
        error(['Field ' nm ' is missing in slayer # ' num2str(num)]);
    end
    [new_fld, res] = validate_type(getfield(slayer,nm), class(getfield(slayer_tmpl,nm)), convert_types);
    if ~res
        error(['Slayer field ' nm ' type missmatch. Should be ' class(getfield(slayer_tmpl,nm))...
            '. Actual ' class(getfield(slayer,nm))]);
    else
        slayer = setfield(slayer,nm,new_fld);
    end
end
out_slayer = slayer;
end