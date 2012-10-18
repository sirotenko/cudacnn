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