#ifndef _TRANSFER_FUNCTIONS_CUH
#define _TRANSFER_FUNCTIONS_CUH
#include "transfer_functions.h"

namespace cudacnn
{

template <class T>
class TansigModCu //: public TransferFunction<T>
{
public:
	__device__ inline T operator() (T x)
	{
		return T(1.7159*tanh(0.66666667*x));
	}
	__device__ inline T dydx(T y)
	{
		T x = T(0.66666667/1.7159*(1.7159+y)*(1.7159-y));
		return x; 

	}
	__device__ inline T d2ydx2(T y)
	{
		T x = T(0.66666667/1.7159*(1.7159+y)*(1.7159-y));
		return x*x; 
	}
};

template <class T>
class PurelinCu //: public TransferFunction<T>
{
public:
	__device__ inline T operator() (T x)
	{
		return x;
	}
	__device__ inline T dydx(T y)
	{
		return 1; 
	}
	__device__ inline T d2ydx2(T y)
	{
		return 1; 
	}
};

template <class T>
class TansigCu //: public TransferFunction<T>
{
public:
	__device__ inline T operator() (T x)
	{
		return tanh(x);
	}
	__device__ inline T dydx(T y)
	{
		return (1-y*y); 
	}
	__device__ inline T d2ydx2(T y)
	{
		return (1-y*y)*(1-y*y); 
	}
};


template <class T>
class SquareCu //: public TransferFunction<T>
{
public:
	__device__ inline T operator() (T x)
	{
		return T(x*x);
	}
	__device__ inline T dydx(T y)
	{
		T x = T(2*y);
		return x; 

	}
	__device__ inline T d2ydx2(T y)
	{
		T x = 0;
		return x*x; 
	}
};

}


#endif
