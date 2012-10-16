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
