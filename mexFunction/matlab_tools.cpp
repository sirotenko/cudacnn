#include "precomp.hpp"
//TODO: Refactor these functions
namespace MatlabTools
{
	//Specializations
	template<> mxClassID MatlabClassID<double>() {return mxDOUBLE_CLASS; }
	template<> mxClassID MatlabClassID<float>() {return mxSINGLE_CLASS; }
	template<> mxClassID MatlabClassID<int>() {return mxINT32_CLASS; }
	template<> mxClassID MatlabClassID<unsigned int>() {return mxUINT32_CLASS; }
    template<> mxClassID MatlabClassID<unsigned _int64>() {return mxUINT64_CLASS; }


	template<>
	bool CheckMatlabType<unsigned char>(mxClassID id)
	{
		if(id == mxUINT8_CLASS) return true;
		return false;
	}

	template<>
	bool CheckMatlabType<double>(mxClassID id)
	{
		if(id == mxDOUBLE_CLASS) return true;
		return false;
	}
	template<>
	bool CheckMatlabType<float>(mxClassID id)
	{
		if(id == mxSINGLE_CLASS) return true;
		return false;
	}

	template<>
	bool CheckMatlabType<int>(mxClassID id)
	{
		if(id == mxINT32_CLASS) return true;
		return false;
	}
	template<>
	bool CheckMatlabType<unsigned int>(mxClassID id)
	{
		if(id == mxUINT32_CLASS) return true;
		return false;
	}
//
//float GetFVal(const mxArray* inp, const char* name)
//{
//	mxArray* tmp = mxGetField(inp,0,name);
//	if(tmp==NULL)
//	{
//		mexPrintf ( "Can't find %s\n", name );
//		mexErrMsgTxt("Unknown version of CNN class");
//	}
//	double* tmp1 = (double*)mxGetData(tmp);
//	return (float)*tmp1;
//}
////Get BYTE field value
//BYTE GetBVal(const mxArray* inp, const char* name)
//{
//	mxArray* tmp = mxGetField(inp,0,name);
//	if(tmp==NULL)
//	{
//		mexPrintf ( "Can't find %s\n", name );
//		mexErrMsgTxt("Unknown version of CNN class");
//	}
//	double* tmp1 = (double*)mxGetData(tmp);
//	return (BYTE)(*tmp1);
//}
////Get UINT field value
//UINT GetUVal(const mxArray* inp, const char* name)
//{
//	mxArray* tmp = mxGetField(inp,0,name);
//	if(tmp==NULL)
//	{
//		mexPrintf ( "Can't find %s\n", name );
//		mexErrMsgTxt("Unknown version of CNN class");
//	}
//	double* tmp1 = (double*)mxGetData(tmp);
//	return (UINT)(*tmp1);
//}
////Get string field value
char* GetSVal(const mxArray* inp, const char* name)
{
	mxArray* tmp = mxGetField(inp,0,name);
	if(tmp==NULL)	{
		std::stringstream ss;
		ss<<"Failed to find field: "<<name;
		throw std::runtime_error(ss.str());
	}
	return mxArrayToString(tmp);
}
//
//template <class T>
//void GetMatOrCell(const mxArray* Wcell, Tensor<T>& out_mat)
//{
//	T* out;
//	size_t m = 0;
//	size_t w = 0;
//	size_t h = 0;
//
//	//Weights may be either cell arrays or matrices, so check it
//	if(mxIsCell(Wcell))
//	{
//		//Weights are cell arrays, so check the number of weight matrices
//		m = mxGetN(Wcell);
//		//Assume that all weight matrices have the same sizes
//		mxArray* Wmat = mxGetCell(Wcell,0);
//		w = mxGetM(Wmat);
//		h = mxGetN(Wmat);
//		size_t matSize = w*h*sizeof(T);
//		//Init weights matrix
//		//out = (float*)mxCalloc( m*w*h, sizeof(float));
//		out = new T[m*w*h];
//		char* ptr = (char*)out;
//		for(size_t i=0;i<m;i++)
//		{
//			mxArray* Wmat = mxGetCell(Wcell,i);
//			memcpy(ptr,mxGetData(Wmat),matSize);
//			ptr+=matSize;
//		}
//	}else
//	{
//		//Init weights matrix
//		m = 1;
//		w = mxGetM(Wcell);
//		h = mxGetN(Wcell);
//		//out = (float*)mxCalloc(w*h,sizeof(float));
//		out = new T[w*h];
//		memcpy(out,mxGetData(Wcell),sizeof(T)*w*h);
//	}
//	std::vector<UINT> dims(3);
//	assert(w < UINT_MAX); assert(h < UINT_MAX); assert(m < UINT_MAX);
//	dims[0] = UINT(w); dims[1] = UINT(h); dims[2] = UINT(m); 
//	out_mat = Tensor<T>(dims, out);
//}
//
//
////Get matrix and it's width and heigth. Function allocates memory so pay attention to freeing it
//void GetMatrix(const mxArray* inp, const char* name, TensorFloat& OutMat)
//{
//
//	//Find the weight field
//	mxArray* Wcell = mxGetField(inp,0,name);
//	if(Wcell==NULL)
//	{
//		mexPrintf ( "Can't find %s\n", name );
//		mexErrMsgTxt("Unknown version of CNN class");
//	}
//	GetMatOrCell(Wcell, OutMat);
//	//OutMat->data_ = out;
//}
//
//void GetMatrix(const mxArray* inp, const char* name, Tensor<int>& OutMat)
//{
//
//	//Find the weight field
//	mxArray* Wcell = mxGetField(inp,0,name);
//	if(Wcell==NULL)
//	{
//		mexPrintf ( "Can't find %s\n", name );
//		mexErrMsgTxt("Unknown version of CNN class");
//	}
//	GetMatOrCell(Wcell, OutMat);
//	//OutMat->data_ = out;
//}
//
//
//========================================  
void SetMatrix(const mxArray* inp, const char* name, const TensorFloat& Mat)
{
	//Find the weight field
	mxArray* Wcell = mxGetField(inp,0,name);
	if(Wcell==NULL)	{
		std::stringstream ss;
		ss<<"Failed to find field: "<<name;
		throw std::runtime_error(ss.str());
	}
	int matSize = Mat.w()*Mat.h();
	//Weights may be either cell arrays or matrices, so check it
	if(mxIsCell(Wcell))
	{
		//Weights are cell arrays, so check the number of weight matrices
		int m = (int)mxGetN(Wcell);
		//Assume that all weight matrices have the same sizes
		//mxArray* Wmat = mxGetCell(Wcell,0);
		mxArray* Wmat = NULL;
		for(int i=0;i<m;i++)
		{
			Wmat = mxGetCell(Wcell,i);
			memcpy(mxGetData(Wmat),Mat.data()+i*matSize,matSize*sizeof(float));
		}
	}else
	{
		memcpy(mxGetData(Wcell),Mat.data(),matSize*sizeof(float));
	}

}

}