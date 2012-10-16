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

#include "common.h"
#include "tensor.h"
#include "exceptions.h"


namespace cudacnn
{

// Wrapper class for operating on GPU
template <class T>
class TensorDev
{
protected:
	size_t num_dims_;
	unsigned* dims_;
	T* data_;
	unsigned num_elements_;

public:
	//This copy constructor
	TensorDev(const TensorGPU<T>& dev_tensor) //This is called in the host
	{
		//Just copy the address, it should be already allocated by cudamalloc
		num_dims_ = dev_tensor.num_dims();
		data_ = dev_tensor.data();
		cutilSafeCall(cudaMalloc(&dims_,sizeof(UINT)*dev_tensor.num_dims()));
		cutilSafeCall(cudaMemcpy(dims_,&dev_tensor.dims()[0],sizeof(UINT)*dev_tensor.num_dims(), cudaMemcpyHostToDevice));
		num_elements_ = dev_tensor.num_elements();
	}
	~TensorDev()
	{
		if(dims_){
			cutilSafeCall(cudaFree(dims_));
			dims_ = NULL;
		}
	}

	__device__ inline T& operator[](int idx)
	{
		assert(idx < num_elements());
		return data_[idx];
	}
	__device__ inline T& operator[](int idx) const 
	{
		assert(idx < num_elements());
		return data_[idx];
	}
	__device__ inline T& operator() (unsigned x, unsigned y, unsigned n)
	{
		assert(num_dims() >= 3);
		assert(x < w() && y < h() && n < d());
		return data_[n*w()*h() + x + y*w()];
	}
	__device__ inline T operator() (unsigned x, unsigned y, unsigned n) const
	{
		assert(num_dims() >= 3);
		assert(x < w() && y < h() && n < d());
		return data_[x + y*w() + n*w()*h()];
	}
	__device__ inline T& operator() (unsigned x, unsigned y)
	{
		assert(num_dims() >= 2);
		assert(x < w() && y < h());
		return data_[x + y*w()];
	}
	__device__ inline T  operator() (unsigned x, unsigned y) const
	{
		assert(num_dims() >= 2);
		assert(x < w() && y < h());
		return data_[x + y*w()];
	}
	__device__ inline T& operator() (unsigned x, unsigned y, unsigned m, unsigned n)
	{
		assert(num_dims() >= 4);
		assert(x < w() && y < h() && m < d() && n < d2());
		return data_[n*d()*w()*h() + m*w()*h() + y*w() + x ];
	}
	__device__ inline T operator() (unsigned x, unsigned y, unsigned m, unsigned n) const
	{
		assert(num_dims() >= 4);
		assert(x < w() && y < h() && m < d() && n < d2());
		return data_[n*d()*w()*h() + m*w()*h() + y*w() + x ];
	}


	__device__ inline T* data() const { return data_; }
	__device__ inline UINT num_dims() const { return num_dims_; }
	__device__ inline int w() const { return num_dims() > 0 ? dims_[0] : 1; }
	__device__ inline int h() const { return num_dims() > 1 ? dims_[1] : 1; }
	__device__ inline int d() const { return num_dims() > 2 ? dims_[2] : 1; }
	__device__ inline int d2() const { return num_dims() > 3 ? dims_[3] : 1; }
	__device__ inline int num_elements() const { return num_elements_; }
private:
	TensorDev(const Tensor<T>& dev_tensor) {};

};

typedef TensorDev<float> TensorDevFloat;
typedef TensorDev<int> TensorDevInt;


template<class T> 
struct TensorDev1
{
	const UINT w; //Width
	const UINT stridex; //Stride along 1st direction
	const size_t num_elements;
	T* data;
	TensorDev1(const TensorGPU<T>& dev_tensor):w(dev_tensor.w()),stridex(1),
		num_elements(dev_tensor.num_elements())//This is called in the host
	{
		//Just copy the address, it should be already allocated by cudamalloc
		data = dev_tensor.data();
		assert(dev_tensor.num_dims() >= 1);
	}
	__device__ inline T& operator[](int idx)
	{
		assert(idx < num_elements);
		return data[idx*stridex];
	}
	__device__ inline T& operator[](int idx) const 
	{
		assert(idx < num_elements);
		return data[idx*stridex];
	}

};

template<class T>
struct TensorDev2 : public TensorDev1<T>
{
	const UINT h;
	const UINT stridey;
	TensorDev2(const TensorGPU<T>& dev_tensor): TensorDev1<T>(dev_tensor), h(dev_tensor.h()),
		stridey(dev_tensor.w())
	{
		assert(dev_tensor.num_dims() >= 2);
	}
	__device__ inline T& operator() (unsigned x, unsigned y)
	{
		assert(x < w && y < h);
		return this->data[x*this->stridex + y*stridey];
	}
	__device__ inline T  operator() (unsigned x, unsigned y) const
	{
		assert(x < w && y < h);
		return this->data[x*this->stridex + y*stridey];
	}
};

template<class T>
struct TensorDev3 : public TensorDev2<T>
{
	const UINT d;
	const UINT striden;
	TensorDev3(const TensorGPU<T>& dev_tensor): TensorDev2<T>(dev_tensor), d(dev_tensor.d()),
		striden(dev_tensor.w()*dev_tensor.h())
	{
		assert(dev_tensor.num_dims() >= 3);
	}
	__device__ inline T& operator() (unsigned x, unsigned y, unsigned n)
	{
		assert(x < w && y < h && n < d);
		return this->data[n*striden + y*this->stridey + x*this->stridex];
	}
	__device__ inline T operator() (unsigned x, unsigned y, unsigned n) const
	{
		assert(x < w && y < h && n < d);
		return this->data[n*striden + y*this->stridey + x*this->stridex];
	}
};

template<class T>
struct TensorDev4 : public TensorDev3<T>
{
	const UINT d2;
	const UINT stridem;
	TensorDev4(const TensorGPU<T>& dev_tensor): TensorDev3<T>(dev_tensor), d2(dev_tensor.d2()),
		stridem(dev_tensor.w()*dev_tensor.h()*dev_tensor.d())
	{
		assert(dev_tensor.num_dims() >= 4);
	}
	__device__ inline T& operator() (unsigned x, unsigned y, unsigned n, unsigned m)
	{
		assert(x < w && y < h && n < d && m < d2);
		return this->data[m*stridem + n*this->striden + y*this->stridey + x*this->stridex];
	}
	__device__ inline T operator() (unsigned x, unsigned y, unsigned n, unsigned m) const
	{
		assert(x < w && y < h && n < d && m < d2);
		return this->data[m*stridem + n*this->striden + y*this->stridey + x*this->stridex];
	}
};
}