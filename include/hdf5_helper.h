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

#ifndef _HDF5_HELPER_H
#define _HDF5_HELPER_H
#define ROOT_LAYERS_GROUP_NAME "/Layers"
#define LAYER_GROUP_NAME "/Layer"
#include <typeinfo>

namespace hdf5Helper
{
	void WriteStringAttribute(const H5::H5Object& obj, const std::string& attrName, const std::string& attrValue);
	void WriteArray(const H5::CommonFG& obj, const std::string& dataName, const cudacnn::TensorFloat* arr);
	void WriteIntAttribute(const H5::H5Object& obj, const std::string& attrName, const int attrValue);
	std::string ReadStringAttribute(const H5::H5Object& obj, const std::string& attrName);
	int ReadIntAttribute(const H5::H5Object& obj, const std::string& attrName);
	void WriteFloatAttribute(const H5::H5Object& obj, const std::string& attrName, const float attrValue);
	float ReadFloatAttribute(const H5::H5Object& obj, const std::string& attrName);

template <class T>
void ReadArrayPT(const H5::CommonFG& obj, const std::string& dataName, cudacnn::Tensor<T>& arr, H5::PredType pt)
{
	H5::DataSet dataset = obj.openDataSet(dataName);

	H5::DataSpace dataspace = dataset.getSpace();
	int rank = dataspace.getSimpleExtentNdims();
	std::vector<hsize_t> dims(rank);
	std::vector<UINT> dims_tensor(rank);
	dataspace.getSimpleExtentDims(&dims[0]);
	for(int i = 0; i < rank; ++i) {
		assert(dims[i] < UINT_MAX);
		//Reverse dims
		dims_tensor[rank - i - 1] = unsigned(dims[i]);
	}
    arr = cudacnn::Tensor<T>(dims_tensor);

	dataset.read(arr.data(), pt);
	//delete[] dims;
}


template <class T>
void ReadArray(const H5::CommonFG& obj, const std::string& dataName, cudacnn::Tensor<T>& arr)
{
	H5::DataSet dataset = obj.openDataSet(dataName);
	H5T_class_t type_class = dataset.getTypeClass();
	if (typeid(T)==typeid(float))
	{
		if( type_class != H5T_FLOAT ) //H5T_ARRAY ?
			throw H5::DataSetIException("ReadArray", "Data should be float");
		ReadArrayPT(obj, dataName, arr, H5::PredType::NATIVE_FLOAT);
	}
	else if(typeid(T)==typeid(double)) {
		if( type_class != H5T_FLOAT ) //Float means both float and double
			throw H5::DataSetIException("ReadArray", "Data should be double");
		ReadArrayPT(obj, dataName, arr, H5::PredType::NATIVE_DOUBLE);
	}
	else if(typeid(T)==typeid(int)) {
		if( type_class != H5T_INTEGER ) 
			throw H5::DataSetIException("ReadArray", "Data should be integer");
		ReadArrayPT(obj, dataName, arr, H5::PredType::NATIVE_INT);
	}
	else if(typeid(T)==typeid(char)) {
		if( type_class != H5T_INTEGER ) 
			throw H5::DataSetIException("ReadArray", "Data should be char");
		ReadArrayPT(obj, dataName, arr, H5::PredType::NATIVE_CHAR);
	}
	else
		throw H5::DataSetIException("ReadArray", "Unknown data type to read");


}
template<class T>
T ReadScalarPT(const H5::CommonFG& obj, const std::string& dataName, H5::PredType pt)
{

	H5::DataSet dataset = obj.openDataSet(dataName);
	H5::DataSpace dataspace = dataset.getSpace();
	int rank = dataspace.getSimpleExtentNdims();
	//if( rank != 1 )
	if( rank > 1 )
		throw H5::DataSetIException("ReadScalarF", "Scalar value should have dimension 1");

	T out;
	dataset.read(&out, pt);
	return out;
}

template <class T>
T ReadScalar(const H5::CommonFG& obj, const std::string& dataName)
{
	H5::DataSet dataset = obj.openDataSet(dataName);
	H5T_class_t type_class = dataset.getTypeClass();
	if (typeid(T)==typeid(float))
	{
		if( type_class != H5T_FLOAT ) 
			throw H5::DataSetIException("ReadScalar", "Data should be float");
		return ReadScalarPT<T>(obj, dataName, H5::PredType::NATIVE_FLOAT);
	}
	else if(typeid(T)==typeid(double)) {
		if( type_class != H5T_FLOAT ) //Float means both float and double
			throw H5::DataSetIException("ReadScalar", "Data should be double");
		return ReadScalarPT<T>(obj, dataName, H5::PredType::NATIVE_DOUBLE);
	}
	else if(typeid(T)==typeid(int)) {
		if( type_class != H5T_INTEGER ) 
			throw H5::DataSetIException("ReadScalar", "Data should be integer");
		return ReadScalarPT<T>(obj, dataName, H5::PredType::NATIVE_INT);
	}
	else if(typeid(T)==typeid(char)) {
		if( type_class != H5T_INTEGER ) 
			throw H5::DataSetIException("ReadScalar", "Data should be char");
		return ReadScalarPT<T>(obj, dataName, H5::PredType::NATIVE_CHAR);
	}
	else
		throw H5::DataSetIException("ReadScalar", "Unknown data type to read");
	//return ReadScalarPT<T>(obj, dataName, pt);

}


}


#endif