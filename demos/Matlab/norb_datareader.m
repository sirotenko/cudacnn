function [data, label, datareader] = norb_datareader(datareader, varargin)
%norb_datareader returns single input and label from NORB database.
%Preprocessing in included. 
%Only one of 2 stereo images is used
%
%  Syntax
%  
%    [data, label, datareader] = norb_datareader(datareader)
%    [data, label, datareader] = norb_datareader(datareader, idx)
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
    if(isfield(datareader, 'buffer_position'))
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

if(~isfield(datareader,'file_pos'))
    datareader.file_pos = idivide(datareader.current,uint32(29160+1)) + 1;
end


%Fill the buffer
%Which file to open
fid_data = fopen(datareader.data_files{datareader.file_pos},'r');  
fid_label = fopen(datareader.label_files{datareader.file_pos},'r');
if(~isfield(datareader,'file_opened') ) %First time open
    magicNum = fread(fid_data,1,'uint32');    %Magic number
    if(magicNum ~= 507333717) 
        error('Error: cant find magic number in data file');
    end
    magicNum = fread(fid_label,1,'uint32');    %Magic number
    if(magicNum ~= 507333716) 
        error('Error: cant find magic number in labels file');
    end
    
    fread(fid_data,1,'uint32');  %Number of dimensions = 4
    datareader.data_num = fread(fid_data,1,'int32');   %Number of samples = 29160
    fread(fid_data,1,'int32');   %views per sample = 2
    datareader.data_w = fread(fid_data,1,'int32');   %width = 108
    datareader.data_h = fread(fid_data,1,'int32');   %height = 108
    fread(fid_label,1,'int32');  %Number of dimensions = 1
    nlabels = fread(fid_label,1,'int32');
    if(datareader.data_num ~= nlabels)
        error('Total number of labels not correspond to the number of inputs');
    end
    
    fread(fid_label,1,'int32');  %Ignore
    fread(fid_label,1,'int32');  %Ignore
    
    %Set the file positions
    datareader.data_start_position = ftell(fid_data); 
    datareader.label_start_position = ftell(fid_label);
    datareader.current = 1;
    datareader.file_opened = true;
end

%Got here if requested out of buffer value

%Reset buffer position
datareader.buffer_position = 1;

data_pos = datareader.data_start_position + ...
    (mod(datareader.current,uint32(datareader.data_num)) - 1)*...
    datareader.data_w*datareader.data_h*2;
label_pos = datareader.label_start_position + ...
    (mod(datareader.current,uint32(datareader.data_num)) - 1);
fseek(fid_data, data_pos, -1);
fseek(fid_label, label_pos, -1);

for k = 1:datareader.buffer_size
    [datareader.data_buffer{k} count] = fread(fid_data,[datareader.data_w ...
        datareader.data_h],'uchar');
    %Stereo pair
    [stereo_pair count] = fread(fid_data,[datareader.data_w datareader.data_h],'uchar');
    if(datareader.stereo)
        datareader.data_buffer{k}(:,:,2) = stereo_pair;
    end

    datareader.data_buffer{k} = uint8(datareader.data_buffer{k});
    if(count == 0) %End of file, switch to next one
        if(datareader.file_pos == length(datareader.data_files))
            %End of last file
            break;
        end
        fclose(fid_data);
        fclose(fid_label);
        datareader.file_pos = datareader.file_pos + 1;
        fid_data = fopen(datareader.data_files{datareader.file_pos},'r');  
        fid_label = fopen(datareader.label_files{datareader.file_pos},'r');
        fseek(fid_data, datareader.data_start_position, -1);        
        fseek(fid_label, datareader.label_start_position, -1);
        %Read one more time
        [datareader.data_buffer{k} count] = fread(fid_data,[datareader.data_w ...
            datareader.data_h],'uchar');        
        if(count == 0) %This shouldn't happen
            error(['Unexpected end of file: ' datareader.data_files{datareader.file_pos}]);
        end        
        %Skip stereo pair
        stereo_pair = fread(fid_data,[datareader.data_w datareader.data_h],'uchar');
        if(datareader.stereo)
            datareader.data_buffer{k}(:,:,2) = stereo_pair;
        end
        datareader.data_buffer{k} = uint8(datareader.data_buffer{k});        
    end
    [datareader.label_buffer{k} count] = fread(fid_label,1,'int');   %Load all labels
    datareader.label_buffer{k} = int32(datareader.label_buffer{k});
    if(count == 0) %This shouldn't happen
        error(['Unexpected end of file: ' datareader.label_files{datareader.file_pos}]);
    end
end

fclose(fid_data);
fclose(fid_label);
%Recursion
[data, label, datareader] = norb_datareader(datareader);
end

function [out_data] = preproc_data(inp_data)
    inp_data = double(inp_data);
    out_data = (inp_data - mean(inp_data(:)))/std(inp_data(:));
end

function [out_label] = preproc_label(inp_label)
    out_label = -ones(1,6);
    out_label(inp_label+1) = 1;
end
