function [data, label, datareader] = mnist_datareader(datareader, varargin)
%mnist_datareader returns single input and label from mnist database.
%Preprocessing in included.
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

if ~isempty(varargin)
    idx = varargin{1};
    if(idx < 1 || idx > datareader.num_samples) 
        error('Index is our of range');
    end
    %New buffer position can be either bigger than buffer size or less than
    %0
    if(isfield(datareader, 'buffer_position '))
        datareader.buffer_position = idx - datareader.current + datareader.buffer_position;
        datareader.current = idx;       
    end
end
%Read from buffer if possible
if(isfield(datareader, 'buffer_position') && ...
    datareader.buffer_position <= datareader.buffer_size &&...
    datareader.buffer_position >= 1 )
    data = datareader.data_buffer{datareader.buffer_position};
    label = datareader.label_buffer{datareader.buffer_position};
    %Preprocess
    data = preproc_data(data);
    label = preproc_label(label);

    datareader.buffer_position = datareader.buffer_position + 1;
    datareader.current = datareader.current + 1;
    return;

end

%Check if we reach end of file
if(isfield(datareader, 'eof') && datareader.eof == true )
    error('No more data available');
end

%Fill the buffer
fid_data = fopen(datareader.data_file,'r','b');  %big-endian
fid_label = fopen(datareader.label_file,'r','b');  %big-endian
if(~isfield(datareader,'data_file_position'))
    magicNum = fread(fid_data,1,'int32');    %Magic number
    if(magicNum~=2051) 
        error('Error: cant find magic number in data file');
    end
    magicNum = fread(fid_label,1,'int32');    %Magic number
    if(magicNum~=2049) 
        error('Error: cant find magic number in labels file');
    end
    
    datareader.data_num = fread(fid_data,1,'int32');  %Number of images
    datareader.row_sz = fread(fid_data,1,'int32');   %Image height
    datareader.col_sz = fread(fid_data,1,'int32');   %Image width
    datareader.lab_num = fread(fid_label,1,'int32');  %Number of labels
    if(datareader.data_num ~= datareader.lab_num)
        error('Total number of labels not correspond to the number of inputs');
    end
    if(datareader.data_num < datareader.num_samples)
        error('Requested number of samples more than actual number of samples in dataset');
    end
 
    
    %Set the file positions
    datareader.data_start_position = ftell(fid_data); 
    datareader.label_start_position = ftell(fid_label);
    datareader.current = 1;
end

%Got here if requested out of buffer value

%Reset buffer position
datareader.buffer_position = 1;
data_pos = datareader.data_start_position + ...
    (datareader.current - 1)*datareader.row_sz*datareader.col_sz;
label_pos = datareader.label_start_position + (datareader.current - 1);
fseek(fid_data, data_pos, -1);
fseek(fid_label, label_pos, -1);

for k = 1:datareader.buffer_size
    [datareader.data_buffer{k} count] = fread(fid_data,[datareader.row_sz ...
        datareader.col_sz],'uchar');
    datareader.data_buffer{k} = uint8(datareader.data_buffer{k});
    if(count == 0)
        datareader.eof = true;
        break;
    end
    [datareader.label_buffer{k} count] = fread(fid_label,1,'uint8');   %Load all labels
    datareader.label_buffer{k} = uint8(datareader.label_buffer{k});
    if(count == 0)
        datareader.eof = true;
        break;
    end    
end
datareader.data_file_position = ftell(fid_data); 
datareader.label_file_position = ftell(fid_label); 

fclose(fid_data);
fclose(fid_label);
%Recursion
[data, label, datareader] = mnist_datareader(datareader);
end


function [out_data] = preproc_data(inp_data)
    out_data = zeros(32,32);
    out_data(3:30,3:30)=double(inp_data);
    out_data = (out_data - mean(out_data(:)))/std(out_data(:));
end

function [out_label] = preproc_label(inp_label)
    out_label = -ones(1,10);
    out_label(inp_label+1) = 1;
end

