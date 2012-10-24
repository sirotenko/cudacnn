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
//TODO: Refactor these functions
namespace MatlabTools
{

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

	template<>
	bool CheckMatlabType<unsigned long long int>(mxClassID id)
	{
		if(id == mxUINT64_CLASS) return true;
		return false;
	}

	template<>
	bool CheckMatlabType<long long int>(mxClassID id)
	{
		if(id == mxINT64_CLASS) return true;
		return false;
	}

std::string GetSVal(const mxArray* inp, const char* name)
{
	mxArray* tmp = mxGetField(inp,0,name);
	if(tmp==NULL)	{
		std::stringstream ss;
		ss<<"Failed to find field: "<<name;
		throw std::runtime_error(ss.str());
	}
    char* mx_string = mxArrayToString(tmp);
    std::string ret_string(mx_string); //Copy constructor
    mxFree(mx_string); //This should be released always
	return ret_string;    
}
//========================================  
void SetMatrix(const mxArray* inp, const char* name, const cudacnn::TensorFloat& Mat)
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
