function [data, label, datareader] = mitfaces_datareader(datareader, varargin)
% mitfaces_datareader returns single input and label from MIT faces dataset.
% It is assumed that these datasets are already preprocessed, and loaded to workspace.
%
%  Syntax
%  
%    [data, label, datareader] = mnist_datareader(datareader)
%    [data, label, datareader] = mnist_datareader(datareader, idx)
%    
%  Description
%   Input:
%    datareader - datareader struct. Contains information about total
%    numberof samples, current sample pointer, etc
%    idx - index of element to read. When ommited, next element is read
%   Output:
%    data - single input
%    label - label for the current input
%
%(c) Sirotenko Mikhail, 2012

if(~isfield(datareader, 'shuffle_idxs'))
    datareader.labels = [ones(datareader.num_faces,1) -1.*ones(datareader.num_faces,1)];
    datareader.indexes = [1:datareader.num_faces 1:datareader.num_nonfaces];
    total_samples = length(datareader.labels);
    datareader.shuffled = ones(total_samples,1)*ceil(rand(total_samples,1)*(total_samples-1));
end
if ~isempty(varargin)
    idx = varargin{1};
else
    idx = datareader.current;
    datareader.current = datareader.current + 1;
end
sh_idx = datareader.shuffled(idx);
label = datareader.labels(sh_idx);
if(label == 1)
    data = datareader.faces{datareader.indexes(sh_idx)};
else
    data = datareader.nonfaces{datareader.indexes(sh_idx)};
end    
