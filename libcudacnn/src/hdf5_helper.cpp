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

#include "precomp.hpp"
#ifdef HAVE_HDF5
//#include "hdf5_helper.h"
using namespace std;
using namespace H5;

namespace hdf5Helper
{


	void WriteStringAttribute(const H5Object& obj, const string& attrName, const string& attrValue)
	{
		StrType attrType(0, H5T_VARIABLE); 
		//StrType attrType;
		DataSpace attrDataspace(H5S_SCALAR);
		Attribute attr = obj.createAttribute(attrName,attrType,attrDataspace);
		attr.write(attrType,attrValue);
	}

	string ReadStringAttribute(const H5Object& obj, const string& attrName)
	{
		StrType attrType(0, H5T_VARIABLE); 
		//StrType attrType;
		Attribute attr = obj.openAttribute(attrName);
		string attrValue;
		attr.read(attrType,attrValue);
		return attrValue;
	}


	void WriteIntAttribute(const H5Object& obj, const string& attrName, const int attrValue)
	{
		//StrType attrType;
		DataSpace attrDataspace(H5S_SCALAR);
		Attribute attr = obj.createAttribute(attrName,PredType::NATIVE_INT,attrDataspace);
		attr.write(PredType::NATIVE_INT,&attrValue);
	}

	int ReadIntAttribute(const H5Object& obj, const string& attrName)
	{
		//StrType attrType;
		Attribute attr = obj.openAttribute(attrName);
		int attrValue = 0;
		attr.read(PredType::NATIVE_INT,&attrValue);
		return attrValue;
	}

	void WriteFloatAttribute(const H5Object& obj, const string& attrName, const float attrValue)
	{
		//StrType attrType;
		DataSpace attrDataspace(H5S_SCALAR);
		Attribute attr = obj.createAttribute(attrName,PredType::NATIVE_FLOAT,attrDataspace);
		attr.write(PredType::NATIVE_INT,&attrValue);
	}

	float ReadFloatAttribute(const H5Object& obj, const string& attrName)
	{
		//StrType attrType;
		Attribute attr = obj.openAttribute(attrName);
		float attrValue = 0;
		attr.read(PredType::NATIVE_FLOAT,&attrValue);
		return attrValue;
	}

	void WriteArray(const CommonFG& obj, const string& dataName, const cudacnn::TensorFloat* arr)
	{
		hsize_t dims[3];
		dims[0] = arr->w();
		dims[1] = arr->h();
		dims[2] = arr->d();

		DataSpace dataspace(3, dims);
		IntType datatype( PredType::NATIVE_FLOAT);
		DataSet dataset = obj.createDataSet( dataName, datatype, dataspace );
		dataset.write( arr->data(), datatype );
	}

	//void ReadArrayF(const CommonFG& obj, const string& dataName, TensorFloat& arr)
	//{
	//	DataSet dataset = obj.openDataSet(dataName);
	//	H5T_class_t type_class = dataset.getTypeClass();
	//	if( type_class != H5T_FLOAT )
	//		throw DataSetIException("ReadArrayF", "Data should be float");

	//	DataSpace dataspace = dataset.getSpace();
	//	int rank = dataspace.getSimpleExtentNdims();
	//	hsize_t* dims = new hsize_t[rank];
	//	unsigned* dims_tensor = new unsigned[rank];
	//	dataset.getArrayType().getArrayDims(dims);
	//	for(int i = 0; i < rank; ++i) {
	//		assert(dims[i] < UINT_MAX);
	//		dims_tensor[i] = unsigned(dims[i]);
	//	}
	//	arr.Init(rank, dims_tensor);

	//	dataset.read(arr.data(), PredType::NATIVE_FLOAT);
	//	delete[] dims;
	//	delete[] dims_tensor;
	//}

	//void ReadArrayI(const CommonFG& obj, const string& dataName, TensorInt& arr)
	//{
	//	DataSet dataset = obj.openDataSet(dataName);
	//	H5T_class_t type_class = dataset.getTypeClass();
	//	if( type_class != H5T_INTEGER )
	//		throw DataSetIException("ReadArrayI", "Data should be integer");

	//	DataSpace dataspace = dataset.getSpace();
	//	int rank = dataspace.getSimpleExtentNdims();
	//	hsize_t* dims = new hsize_t[rank];
	//	unsigned* dims_tensor = new unsigned[rank];
	//	dataset.getArrayType().getArrayDims(dims);
	//	for(int i = 0; i < rank; ++i) {
	//		assert(dims[i] < UINT_MAX);
	//		dims_tensor[i] = unsigned(dims[i]);
	//	}
	//	arr.Init(rank, dims_tensor);

	//	dataset.read(arr.data(), PredType::NATIVE_INT);
	//	delete[] dims;
	//	delete[] dims_tensor;
	//}

	//float ReadScalarF(const CommonFG& obj, const string& dataName)
	//{

	//	DataSet dataset = obj.openDataSet(dataName);
	//	H5T_class_t type_class = dataset.getTypeClass();
	//	if( type_class != H5T_FLOAT )
	//		throw DataSetIException("ReadScalarF", "Data should be float");
	//	DataSpace dataspace = dataset.getSpace();
	//	int rank = dataspace.getSimpleExtentNdims();
	//	if( rank != 1 )
	//		throw DataSetIException("ReadScalarF", "Scalar value should have dimension 1");
	//		
	//	float out;
	//	dataset.read(&out, PredType::NATIVE_FLOAT);
	//	return out;
	//}

	//int ReadScalarI(const CommonFG& obj, const string& dataName)
	//{
	//	DataSet dataset = obj.openDataSet(dataName);
	//	H5T_class_t type_class = dataset.getTypeClass();
	//	if( type_class != H5T_INTEGER )
	//		throw DataSetIException("ReadScalarI", "Data should be integer");
	//	DataSpace dataspace = dataset.getSpace();
	//	int rank = dataspace.getSimpleExtentNdims();
	//	if( rank != 1 )
	//		throw DataSetIException("ReadScalarF", "Scalar value should have dimension 1");

	//	int out;
	//	dataset.read(&out, PredType::NATIVE_INT);
	//	return out;
	//}
}
#endif //HAVE_HDF5