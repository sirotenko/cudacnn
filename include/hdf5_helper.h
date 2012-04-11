#pragma once

#ifndef _HDF5_HELPER_H
#define _HDF5_HELPER_H
namespace hdf5Helper
{
	void WriteStringAttribute(const H5::H5Object& obj, const std::string& attrName, const std::string& attrValue);
	void WriteArray(const H5::CommonFG& obj, const std::string& dataName, const TensorFloat* arr);
	void WriteIntAttribute(const H5::H5Object& obj, const std::string& attrName, const int attrValue);
	std::string ReadStringAttribute(const H5::H5Object& obj, const std::string& attrName);
	int ReadIntAttribute(const H5::H5Object& obj, const std::string& attrName);
	void WriteFloatAttribute(const H5::H5Object& obj, const std::string& attrName, const float attrValue);
	float ReadFloatAttribute(const H5::H5Object& obj, const std::string& attrName);
	//void ReadArrayF(const H5::CommonFG& obj, const std::string& dataName, TensorFloat& arr);
	//void ReadArrayI(const H5::CommonFG& obj, const std::string& dataName, TensorInt& arr);
	//float ReadScalarF(const H5::CommonFG& obj, const std::string& dataName);
	//int ReadScalarI(const H5::CommonFG& obj, const std::string& dataName);
	//void ReadArrayPT(const H5::CommonFG& obj, const std::string& dataName, TensorInt& arr, PredType pt);


template <class T>
void ReadArrayPT(const H5::CommonFG& obj, const std::string& dataName, Tensor<T>& arr, H5::PredType pt)
{
	DataSet dataset = obj.openDataSet(dataName);

	DataSpace dataspace = dataset.getSpace();
	int rank = dataspace.getSimpleExtentNdims();
	std::vector<hsize_t> dims(rank);
	std::vector<UINT> dims_tensor(rank);
	dataspace.getSimpleExtentDims(&dims[0]);
	for(int i = 0; i < rank; ++i) {
		assert(dims[i] < UINT_MAX);
		//Reverse dims
		dims_tensor[rank - i - 1] = unsigned(dims[i]);
	}
	arr = Tensor<T>(dims_tensor);

	dataset.read(arr.data(), pt);
	//delete[] dims;
}


template <class T>
void ReadArray(const H5::CommonFG& obj, const std::string& dataName, Tensor<T>& arr)
{
	DataSet dataset = obj.openDataSet(dataName);
	H5T_class_t type_class = dataset.getTypeClass();
	if (typeid(T)==typeid(float))
	{
		if( type_class != H5T_FLOAT ) //H5T_ARRAY ?
			throw DataSetIException("ReadArray", "Data should be float");
		ReadArrayPT(obj, dataName, arr, PredType::NATIVE_FLOAT);
	}
	else if(typeid(T)==typeid(double)) {
		if( type_class != H5T_FLOAT ) //Float means both float and double
			throw DataSetIException("ReadArray", "Data should be double");
		ReadArrayPT(obj, dataName, arr, PredType::NATIVE_DOUBLE);
	}
	else if(typeid(T)==typeid(int)) {
		if( type_class != H5T_INTEGER ) 
			throw DataSetIException("ReadArray", "Data should be integer");
		ReadArrayPT(obj, dataName, arr, PredType::NATIVE_INT);
	}
	else if(typeid(T)==typeid(char)) {
		if( type_class != H5T_INTEGER ) 
			throw DataSetIException("ReadArray", "Data should be char");
		ReadArrayPT(obj, dataName, arr, PredType::NATIVE_CHAR);
	}
	else
		throw DataSetIException("ReadArray", "Unknown data type to read");


}
template<class T>
T ReadScalarPT(const H5::CommonFG& obj, const std::string& dataName, H5::PredType pt)
{

	DataSet dataset = obj.openDataSet(dataName);
	DataSpace dataspace = dataset.getSpace();
	int rank = dataspace.getSimpleExtentNdims();
	//if( rank != 1 )
	if( rank > 1 )
		throw DataSetIException("ReadScalarF", "Scalar value should have dimension 1");

	T out;
	dataset.read(&out, pt);
	return out;
}

template <class T>
T ReadScalar(const H5::CommonFG& obj, const std::string& dataName)
{
	DataSet dataset = obj.openDataSet(dataName);
	H5T_class_t type_class = dataset.getTypeClass();
	if (typeid(T)==typeid(float))
	{
		if( type_class != H5T_FLOAT ) 
			throw DataSetIException("ReadScalar", "Data should be float");
		return ReadScalarPT<T>(obj, dataName, PredType::NATIVE_FLOAT);
	}
	else if(typeid(T)==typeid(double)) {
		if( type_class != H5T_FLOAT ) //Float means both float and double
			throw DataSetIException("ReadScalar", "Data should be double");
		return ReadScalarPT<T>(obj, dataName, PredType::NATIVE_DOUBLE);
	}
	else if(typeid(T)==typeid(int)) {
		if( type_class != H5T_INTEGER ) 
			throw DataSetIException("ReadScalar", "Data should be integer");
		return ReadScalarPT<T>(obj, dataName, PredType::NATIVE_INT);
	}
	else if(typeid(T)==typeid(char)) {
		if( type_class != H5T_INTEGER ) 
			throw DataSetIException("ReadScalar", "Data should be char");
		return ReadScalarPT<T>(obj, dataName, PredType::NATIVE_CHAR);
	}
	else
		throw DataSetIException("ReadScalar", "Unknown data type to read");
	//return ReadScalarPT<T>(obj, dataName, pt);

}


}


#endif