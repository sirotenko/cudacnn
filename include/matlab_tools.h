//Copyright (c) 2012, Mikhail Sirotenko <mihail.sirotenko@gmail.com>
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef __MATLAB_TOOLS_H
#define __MATLAB_TOOLS_H



namespace MatlabTools
{


/*//Convert c++ type to matlab class ID
template<class T>	mxClassID MatlabClassID();
//Default implementation
template<class T>	mxClassID MatlabClassID() {
	std::stringstream ss;
	ss<<"Unsupported type, sizeof = ";
	ss<<sizeof(T);
	throw std::runtime_error(ss.str());
};*/

//! Generic helper for matlab_get_class
/*! If there are no specialized matches for __Data then we say that the ClassID that corresponds to this data type is unknown */
template<class __Data>
struct matlab_get_class_helper
{
    const static mxClassID value = mxUNKNOWN_CLASS;
};	
//! Generic helper for matlab_get_type
/*! If there are no specialized matches for __ClassId__ then we say that the type that corresponds to this ClassID is unknown */
template<unsigned int __ClassId__>
struct matlab_get_type_helper
{
    typedef struct UnknownType type;
};	
// macro to declare a two-way type conversion between __Type and CLASS
// specifically, we make a specialized matlab_get_class_helper instance to look up CLASS given __Type
// and a specialized matlab_get_type_helper to look up __Type given CLASS
#define DECLARE_MATLAB_TYPE_CONVERSION( __Type, CLASS ) \
template<> \
struct matlab_get_class_helper<__Type> \
{ \
    const static mxClassID value = CLASS; \
}; \
 \
template<> \
struct matlab_get_type_helper<CLASS> \
{ \
    typedef __Type type; \
}	
// declare conversions for known types
DECLARE_MATLAB_TYPE_CONVERSION( bool, mxLOGICAL_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( char, mxCHAR_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( void, mxVOID_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( double, mxDOUBLE_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( float, mxSINGLE_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( int8_t, mxINT8_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( uint8_t, mxUINT8_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( int16_t, mxINT16_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( uint16_t, mxUINT16_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( int32_t, mxINT32_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( uint32_t, mxUINT32_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( int64_t, mxINT64_CLASS );
DECLARE_MATLAB_TYPE_CONVERSION( uint64_t, mxUINT64_CLASS );	
//! Defines the mxClassID that corresponds to the __Data
template<class __Data>
struct matlab_get_class
{
    const static mxClassID value = matlab_get_class_helper<__Data>::value;
};	
//! Defines the data type that corresponds to __ClassId__
template<unsigned int __ClassId__>
struct matlab_get_type
{
    typedef typename matlab_get_type_helper<__ClassId__>::type type;
};

template<class T> bool CheckMatlabType(mxClassID id);

std::string GetSVal(const mxArray* inp, const char* name);
template <class T>
void GetMatOrCell(const mxArray* Wcell, cudacnn::Tensor<T>& out_mat);

template<class T>
T GetScalar(const mxArray* scal)
{
	if (!mxIsNumeric(scal)) {
		std::stringstream ss;
		ss<<"Scalar should be numeric";
		throw std::runtime_error(ss.str());
	}
	if (mxIsEmpty(scal)) {
		std::stringstream ss;
		ss<<"Scalar should be non empty";
		throw std::runtime_error(ss.str());
	}

	if(  mxGetNumberOfElements(scal)!= 1 ){
		std::stringstream ss;
		ss<<"Scalar should be a scalar";
		throw std::runtime_error(ss.str());
	}
	void* pdata = mxGetData(scal);
	T data;
	mxClassID id = mxGetClassID(scal);
	switch(id){
		case mxDOUBLE_CLASS:
			data = static_cast<T>(*static_cast<double*>(pdata)); break;
		case mxSINGLE_CLASS:
			data = static_cast<T>(*static_cast<float*>(pdata)); break;
		case mxINT8_CLASS:
			data = static_cast<T>(*static_cast<char*>(pdata)); break;
		case mxUINT8_CLASS:
			data = static_cast<T>(*static_cast<unsigned char*>(pdata)); break;
		case mxINT16_CLASS:
			data = static_cast<T>(*static_cast<short*>(pdata)); break;
		case mxUINT16_CLASS:
			data = static_cast<T>(*static_cast<unsigned short*>(pdata)); break;
		case mxINT32_CLASS:
			data = static_cast<T>(*static_cast<int*>(pdata)); break;
		case mxUINT32_CLASS:
			data = static_cast<T>(*static_cast<UINT*>(pdata)); break;
		case mxINT64_CLASS:
			data = static_cast<T>(*static_cast<long long*>(pdata)); break;
		case mxUINT64_CLASS:
			data = static_cast<T>(*static_cast<unsigned long long*>(pdata)); break;
		default:
			std::stringstream ss;
			ss<<". Unknown or unsupported scalar data type.";
			throw std::runtime_error(ss.str());

	}
	return data;
}

template<class T>
T GetScalar(const mxArray* inp, const char* name)
{
	mxArray* scal = mxGetField(inp,0,name);
	if(scal == NULL){
		std::stringstream ss;
		ss<<name<<" not found.";
		throw std::runtime_error(ss.str());
	}
	try{
		return GetScalar<T>(scal);
	}
	catch(std::runtime_error& ex)	{
		std::stringstream ss;
		ss<<"Failed to load "<<name<<std::endl;
		ss<<ex.what();
		throw std::runtime_error(ss.str());
	}

}

template<class T>
void SetScalar(T data, const char* name, mxArray* inp)
{
	mxArray* scal = mxGetField(inp,0,name);
	if(scal == NULL){
		std::stringstream ss;
		ss<<name<<" not found.";
		throw std::runtime_error(ss.str());
	}
	if (!mxIsNumeric(scal)) {
		std::stringstream ss;
		ss<<name<<" should be numeric";
		throw std::runtime_error(ss.str());
	}
	if (mxIsEmpty(scal)) {
		std::stringstream ss;
		ss<<name<<" should be non empty";
		throw std::runtime_error(ss.str());
	}

	if(  mxGetNumberOfElements(scal)!= 1 ){
		std::stringstream ss;
		ss<<name<<" should be a scalar";
		throw std::runtime_error(ss.str());
	}
	void* pdata = mxGetData(scal);
	mxClassID id = mxGetClassID(scal);
	switch(id){
		case mxDOUBLE_CLASS:
			*pdata = static_cast<double>(data); break;
		case mxSINGLE_CLASS:
			*pdata = static_cast<float>(data); break;
		case mxINT8_CLASS:
			*pdata = static_cast<char>(data); break;
		case mxUINT8_CLASS:
			*pdata = static_cast<unsigned char>(data); break;
		case mxINT16_CLASS:
			*pdata = static_cast<short>(data); break;
		case mxUINT16_CLASS:
			*pdata = static_cast<unsigned short>(data); break;
		case mxINT32_CLASS:
			*pdata = static_cast<int>(data); break;
		case mxUINT32_CLASS:
			*pdata = static_cast<UINT>(data); break;
		case mxINT64_CLASS:
			*pdata = static_cast<long long>(data); break;
		case mxUINT64_CLASS:
			*pdata = static_cast<unsigned long>(data); break;
		default:
			std::stringstream ss;
			ss<<"Failed to load "<<name<<". Unknown or unsupported scalar data type.";
			throw std::runtime_error(ss.str());
	}
}



template<class T>
void AddScalar(T data, const char* name, mxArray* inp)
{
	mxArray* scal = mxGetField(inp,0,name);
	if(scal != NULL){
		std::stringstream ss;
		ss<<"Failed to add field. "<<name<<" already exist.";
		throw std::runtime_error(ss.str());
	}
	if(!mxIsStruct(inp)){
		std::stringstream ss;
		ss<<"Failed to add field. Input array should be struct";
		throw std::runtime_error(ss.str());
	}
	mxAddField(inp,name);

	mxArray* data_arr = mxCreateNumericMatrix(1,1,matlab_get_class<T>::value,mxREAL);

	T* pdata = static_cast<T*>(mxGetData(data_arr));
	*pdata = data;
	mxSetField(inp,0,name,data_arr);
}


template<class T>
void GetArray(const mxArray* arr, cudacnn::Tensor<T>& out_tens);


template<class T>
void GetArray(const mxArray* inp, const char* name, cudacnn::Tensor<T>& out_tens)
{
	mxArray* arr = mxGetField(inp,0,name);
	if(arr == NULL){
		std::stringstream ss;
		ss<<name<<" not found.";
		throw std::runtime_error(ss.str());
	}
	try{
		GetArray<T>(arr, out_tens);
	}
	catch(std::runtime_error& ex){
		std::stringstream ss;
		ss<<ex.what()<<": "<<name;
		throw std::runtime_error(ss.str());
	}
}
template<class T>
void GetArray(const mxArray* inp, const char* name, cudacnn::TensorGPU<T>& out_tens)
{
	cudacnn::Tensor<T> tmp;
	GetArray(inp, name, tmp);
	out_tens = tmp;
}

template<class T>
void GetArray(const mxArray* arr, cudacnn::Tensor<T>& out_tens)
{
	if (!mxIsNumeric(arr)) {
		std::stringstream ss;
		ss<<"Array should be numeric";
		throw std::runtime_error(ss.str());
	}
	if (mxIsEmpty(arr)) {
		std::stringstream ss;
		ss<<"Array should be non empty";
		throw std::runtime_error(ss.str());
	}

	mxClassID id = mxGetClassID(arr);
	if(!CheckMatlabType<T>(id)){
		std::stringstream ss;
		ss<<"Failed to load Array. Type mismatch.";
		throw std::runtime_error(ss.str());
	}
	T* pdata = static_cast<T*>(mxGetData(arr));
	size_t ndims = mxGetNumberOfDimensions(arr);
	std::vector<UINT> dims(ndims);
	const mwSize* dims_ptr = mxGetDimensions(arr);
	for(UINT i = 0; i < ndims; ++i) dims[i] = static_cast<UINT>(dims_ptr[i]);
	out_tens = cudacnn::Tensor<T>(dims, pdata);
}

template<class T>
void GetArray(const mxArray* arr, cudacnn::TensorGPU<T>& out_tens)
{
	cudacnn::Tensor<T> tmp;
	GetArray(arr, tmp);
	out_tens = tmp;
}

template<class T>
void AddArray(const cudacnn::Tensor<T>& data, const char* name, mxArray* inp);


template<class T>
void AddArray(const cudacnn::TensorGPU<T>& data, const char* name, mxArray* inp)
{
	//Copy from GPU to host memory
	cudacnn::Tensor<T> data_host = data;
	AddArray(data_host, name, inp);
}

template<class T>
void AddArray(const cudacnn::Tensor<T>& data, const char* name, mxArray* inp)
{
	mxArray* scal = mxGetField(inp,0,name);
	if(scal != NULL){
		std::stringstream ss;
		ss<<"Failed to add field. "<<name<<" already exist.";
		throw std::runtime_error(ss.str());
	}
	if(!mxIsStruct(inp)){
		std::stringstream ss;
		ss<<"Failed to add field. Input array should be struct";
		throw std::runtime_error(ss.str());
	}
	mxAddField(inp,name);
	size_t ndims = data.num_dims();
	const std::vector<UINT>& dims = data.dims();
	mwSize* dims_ptr = new mwSize[ndims];
	for(UINT i = 0; i < ndims; ++i) {
		dims_ptr[i] = static_cast<mwSize>(dims[i]);
	}

	mxArray* data_arr = mxCreateNumericArray(mwSize(ndims),dims_ptr,matlab_get_class<T>::value,mxREAL);
	T* pdata = static_cast<T*>(mxGetData(data_arr));
	memcpy(pdata, data.data(), data.num_elements()*sizeof(T));
	mxSetField(inp,0,name,data_arr);
}


// arr should be properly initialized
template<class T>
void SetArray(const cudacnn::Tensor<T>& inp, mxArray* arr)
{
	if (!mxIsNumeric(arr)) {
		std::stringstream ss;
		ss<<"Array should be numeric";
		throw std::runtime_error(ss.str());
	}
	if (mxIsEmpty(arr)) {
		std::stringstream ss;
		ss<<"Array should be non empty";
		throw std::runtime_error(ss.str());
	}

	mxClassID id = mxGetClassID(arr);
	if(!CheckMatlabType<T>(id)){
		std::stringstream ss;
		ss<<"Failed to save Array. Type mismatch.";
		throw std::runtime_error(ss.str());
	}
	T* pdata = static_cast<T*>(mxGetData(arr));
	size_t ndims = mxGetNumberOfDimensions(arr);
	if(static_cast<UINT>(ndims) != inp.ndims()) {
		throw std::runtime_error("Failed to save array. Ndims mismatch.");
	}
	UINT* dims = inp.dims();
	const mwSize* dims_ptr = mxGetDimensions(arr);
	for(int i = 0; i < ndims; ++i) {
		if(dims[i] != static_cast<UINT>(dims_ptr[i])) {
			std::stringstream ss;
			ss<<"Failed to save Array. Dimension mismatch.";
			ss<<"Dim # "<<i<<"size for input arr = "<<dims[i]<<"; for mxArray = "<<dims_ptr[i]<<std::endl;
			throw std::runtime_error(ss.str());
		}
	}
	memcpy(pdata, inp.data(), inp.num_elements()*sizeof(T));
}

template<class T>
void SetArray(const cudacnn::Tensor<T>& inp, mxArray* struct_arr, const char* name)
{
	mxArray* arr = mxGetField(struct_arr,0,name);
	if(arr == NULL){
		std::stringstream ss;
		ss<<name<<" not found.";
		throw std::runtime_error(ss.str());
	}
	try{
		SetArray<T>(inp, arr);
	}
	catch(std::runtime_error& ex){
		std::stringstream ss;
		ss<<ex.what()<<": "<<name;
		throw std::runtime_error(ss.str());
	}

}

}

#endif
