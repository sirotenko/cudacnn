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

#ifndef _TRANSFER_FUNCTIONS_H
#define _TRANSFER_FUNCTIONS_H

namespace cudacnn
{

enum eTransfFunc
{
	eTransferUnknown,
	ePurelin,
	eTansig_mod,
	eTansig,
	eShrinkage,
	eSquare
};

template <class T>
class TansigMod
{
public:
	__device__ T operator() (T x)
	{
		return T(1.7159*tanh(0.66666667*x));
	}
	__device__ T dydx(T fn_out)
	{
		T x = T(0.66666667/1.7159*(1.7159+fn_out)*(1.7159-fn_out));
		return x; 

	}
	__device__ T d2fn_outdx2(T fn_out)
	{
		T x = T(0.66666667/1.7159*(1.7159+fn_out)*(1.7159-fn_out));
		return x*x; 
	}
	std::string name() const { return "tansig_mod";}
};

template <class T>
class Purelin //: public TransferFunctionBase<T>
{
public:
	__device__ T operator() (T x)
	{
		return x;
	}
	__device__ T dydx(T fn_out)
	{
		return 1; 
	}
	__device__ T d2ydx2(T fn_out)
	{
		return 0; 
	}
	std::string name() const { return "purelin";}
};

template <class T>
class Tansig //: public TransferFunctionBase<T>
{
public:
	__device__ T operator() (T x)
	{
		return tanh(x);
	}
	__device__ T dydx(T fn_out)
	{
		return (1-fn_out*fn_out); 
	}
	__device__ T d2ydx2(T fn_out)
	{
		return (1-fn_out*fn_out)*(1-fn_out*fn_out); 
	}
	std::string name() const { return "tansig";}
};

}

#endif
