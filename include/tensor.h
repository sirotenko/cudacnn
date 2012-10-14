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

#ifndef _TENSOR_H
#define _TENSOR_H

namespace cudacnn
{
template <class T> class Tensor;
template <class T> class TensorGPU;

//Tensor interface
template <class T> 
class BTensor {
public:
	typedef T element_type;
	// Constructors
	BTensor(): 	data_(NULL), shallow_(false), dims_(0) {};
	//Pure virtual 
	virtual void ZeroMemory() = 0;
	//Utility
	virtual bool HaveSameSize(const BTensor<T> &tens) const;
	// Getters
	inline T* data() const { return data_; }
	inline size_t num_dims() const { return dims_.size(); }
	inline const std::vector<unsigned>& dims() const { return dims_; }
	inline unsigned num_elements() const;
	inline unsigned w() const { return num_dims() > 0 ? dims_[0] : 0; }
	inline unsigned h() const { return num_dims() > 1 ? dims_[1] : 1; }
	inline unsigned d() const { return num_dims() > 2 ? dims_[2] : 1; }
	inline unsigned d2() const { return num_dims() > 3 ? dims_[3] : 1; }

	virtual void Reshape(const std::vector<unsigned>& new_dims);
    //Convert multidimensional tensor to vector
    virtual void Flatten();
protected:
	virtual void Destroy() = 0;// { if(!shallow_)	DeallocateMemory();};
	//Inplementation dependent
	virtual void AllocateMemory() = 0;
	virtual void DeallocateMemory() = 0;

	std::vector<unsigned> dims_;
	T* data_;
	//Flag indicating is it necessary to deallocate data in Destroy
	bool shallow_;
};

//==================  Getters  ===============
template<class T>
unsigned BTensor<T>::num_elements() const
{
	std::vector<unsigned>::const_iterator it;
	unsigned numel;
	if (dims_.size() == 0) {
		numel = 0;
	} else {
		numel = 1;
		for (it = dims_.begin(); it!= dims_.end(); ++it)	{
			numel *= *it;
		}
	}
	return numel;
}

//================  Utility ======================
template<class T>
bool BTensor<T>::HaveSameSize(const BTensor<T> &tens) const
{
	if(tens.num_dims() != num_dims()) return false;
	for (UINT i = 0; i < num_dims(); ++i){
		if(dims()[i] != tens.dims()[i]) return false;
	}
	return true;
}

//TODO: consider specialization of this function for GPU in case of pitched memory allocation
template<class T>
void BTensor<T>::Reshape(const std::vector<unsigned>& new_dims)
{
	std::vector<unsigned>::const_iterator it;
	unsigned numel = 1;
	for (it = new_dims.begin(); it!= new_dims.end(); ++it)	{
		numel *= *it;
	}
	if(numel != num_elements()) 
		throw std::runtime_error("Failed to do tensor reshape. Number of elements not correspond to new dimensions");
	dims_ = new_dims;
}

template<class T>
void BTensor<T>::Flatten()
{
    UINT nelem = num_elements();
    dims_ = std::vector<UINT>(1);
    dims_[0] = nelem; 
}

template <class T> 
class Tensor : public BTensor<T>
{
public:

	virtual ~Tensor() {  Destroy();  }
	Tensor() {};  //Default ctor
	//ctors
	Tensor(const std::vector<unsigned>& dims_in);
	Tensor(int iw, int ih, int im);
	Tensor(const std::vector<unsigned>& dims_in, T* data);

	Tensor(const TensorGPU<T>& tens) {	*this = tens;	}
	Tensor(const Tensor<T>& tens, bool shallow = false);

	/* Use input matrix dimension to initialize, but not data */
	virtual void ZeroMemory();
	static Tensor<T> Ones(const std::vector<unsigned>& dims_in);
	static Tensor<T> Rand(const std::vector<unsigned>& dims_in, T sigma);
	//Operators
	Tensor<T>& operator = (const Tensor<T> &rhs);
	Tensor<T>& operator = (const TensorGPU<T> &rhs);
	Tensor<T> operator - (const Tensor<T> &rhs) const;
	inline T& operator [](unsigned idx);
	inline const T& operator [](unsigned idx) const;
	inline T& operator() (unsigned x, unsigned y, unsigned m, unsigned n);
	inline T  operator() (unsigned x, unsigned y, unsigned m, unsigned n) const;
	inline T& operator() (unsigned x, unsigned y, unsigned n);
	inline T  operator() (unsigned x, unsigned y, unsigned n) const;
	inline T&  operator() (unsigned x, unsigned y);
	inline T  operator() (unsigned x, unsigned y) const;

protected:
	virtual void Destroy() { if(!shallow_)	DeallocateMemory();};
	virtual void AllocateMemory();
	virtual void DeallocateMemory();
};


typedef Tensor<float> TensorFloat;
typedef Tensor<double> TensorDouble;
typedef Tensor<int> TensorInt;

//==============  Constructors ======================
//Init tensor as a wrapper for the data
template <class T>
Tensor<T>::Tensor(const std::vector<unsigned>& dims_in, T* data_in)
{
	shallow_ = true;
	dims_ = dims_in;
	data_ = data_in;
}

template<class T>
Tensor<T>::Tensor(const std::vector<unsigned>& dims_in)
{
	dims_ = dims_in;
	AllocateMemory();
}


template<class T>
Tensor<T>::Tensor(int iw, int ih, int im)
{
	std::vector<unsigned> in_dims(3,0);
	in_dims[0] = iw;
	in_dims[1] = ih;
	in_dims[2] = im;
	dims_ = in_dims;
	AllocateMemory();
}

template<class T>
Tensor<T>::Tensor(const Tensor<T>& tens, bool shallow)
{
    if(!shallow){
        *this = tens;
    }else {
        data_ = tens.data_;
        dims_ = tens.dims_;
        shallow_ = shallow; //True
    }
    
}

template<class T>
void Tensor<T>::AllocateMemory()
{
	data_ = new T[num_elements()];
	memset(data_, 0, num_elements()*sizeof(T));
}

template<class T>
void Tensor<T>::DeallocateMemory()
{
	delete[] data_; data_ = NULL;
}

template<class T>
void Tensor<T>::ZeroMemory()
{
	memset(data_, 0, num_elements()*sizeof(T));
}

template<class T>
Tensor<T> Tensor<T>::Ones(const std::vector<unsigned>& dims_in)
{
	Tensor<T> tens(dims_in);
	for (UINT i = 0; i < tens.num_elements(); ++i)	{
		tens[i] = T(1);
	}
	return tens;
}

template <class T>
Tensor<T> Tensor<T>::Rand(const std::vector<unsigned>& dims_in, T sigma )
{
	Tensor<T> tens(dims_in);
	for (UINT i = 0; i < tens.num_elements(); ++i)	{
		tens[i] = static_cast<T>(-0.5 + (double(std::rand()) / RAND_MAX))*sigma;
	}
	return tens;
}

template<class T>
Tensor<T>& Tensor<T>::operator = (const Tensor<T> &rhs)
{
	if (this == &rhs)      // Same object?
		return *this;
	Destroy();
	dims_ = rhs.dims();
	AllocateMemory();
	memcpy(data_,rhs.data(),sizeof(T)*num_elements());
	return *this;
}

template<class T>
Tensor<T>& Tensor<T>::operator=(const TensorGPU<T> &rhs)
{
	Destroy();
	dims_ = rhs.dims();
	AllocateMemory();
	cutilSafeCall(cudaMemcpy(data_,rhs.data(),sizeof(T)*num_elements(), cudaMemcpyDeviceToHost));
	return *this;
}

template<class T>
Tensor<T> Tensor<T>::operator - (const Tensor<T> &rhs) const
{
	assert(HaveSameSize(rhs));
	Tensor<T> out_tens(rhs);
	for(UINT i = 0; i < num_elements(); ++i){
		out_tens[i] = data()[i] - rhs[i];
	}
	return out_tens;
}
template<class T>
inline T& Tensor<T>::operator [](unsigned idx)
{
	assert(idx < num_elements());
	return data_[idx];
}
template<class T>
inline const T& Tensor<T>::operator [](unsigned idx) const
{
	assert(idx < num_elements());
	return data_[idx];
}

template<class T>
inline T& Tensor<T>::operator() (unsigned x, unsigned y, unsigned m, unsigned n)
{
	assert(num_dims() <= 4);
	assert(x < w() && y < h() && m < d() && n < d2());
	return data_[n*d()*w()*h() + m*w()*h() + y*w() + x ];
}

template<class T>
inline T  Tensor<T>::operator() (unsigned x, unsigned y, unsigned m, unsigned n) const
{
	assert(num_dims() <= 4);
	assert(x < w() && y < h() && m < d() && n < d2());
	return data_[n*d()*w()*h() + m*w()*h() + y*w() + x ];
}


template<class T>
inline T& Tensor<T>::operator() (unsigned x, unsigned y, unsigned n)
{
	assert(num_dims() <= 3);	
	assert(x < w() && y < h() && n < d());
	return data_[n*w()*h() + x + y*w()];
}

template<class T>
inline T  Tensor<T>::operator() (unsigned x, unsigned y, unsigned n) const
{
	assert(num_dims() <= 3);
	assert(x < w() && y < h() && n < d());
	return data_[n*w()*h() + x + y*w()];
}
template<class T>
inline T&  Tensor<T>::operator() (unsigned x, unsigned y)
{
	assert(num_dims() <= 2);
	assert(x < w() && y < h());
	return data_[x + y*w()];
}
template<class T>
inline T  Tensor<T>::operator() (unsigned x, unsigned y) const
{
	assert(num_dims() <= 2);
	assert(x < w() && y < h());
	return data_[x + y*w()];
}

template <class T>
class TensorGPU : public BTensor<T>
{
public:

	TensorGPU() {};
	TensorGPU(const std::vector<unsigned>& dims_in);
	TensorGPU(int iw, int ih, int im);
	TensorGPU(const std::vector<unsigned>& dims_in, T* data);

	TensorGPU(const Tensor<T>& tens){	*this = tens;	}
	TensorGPU(const TensorGPU<T>& tens, bool shallow = false);

	virtual ~TensorGPU() { 	Destroy(); 	}
	virtual void ZeroMemory();
	TensorGPU<T>& operator =(const TensorGPU<T>& rhs );
	TensorGPU<T>& operator =(const Tensor<T>& rhs );
	inline T& operator [](unsigned idx);
	inline const T& operator [](unsigned idx) const;
	inline T& operator() (unsigned x, unsigned y, unsigned n);
	inline T  operator() (unsigned x, unsigned y, unsigned n) const;
protected:
	virtual void Destroy() { if(!shallow_)	DeallocateMemory();};
	virtual void AllocateMemory();
	virtual void DeallocateMemory();
};

typedef TensorGPU<float> TensorGPUFloat;
typedef TensorGPU<int> TensorGPUInt;


template <class T>
TensorGPU<T>::TensorGPU(const std::vector<unsigned>& dims_in, T* data_in)
{
	shallow_ = true;
	dims_ = dims_in;
	data_ = data_in;
}

template<class T>
TensorGPU<T>::TensorGPU(const std::vector<unsigned>& dims_in)
{
	dims_ = dims_in;
	AllocateMemory();
}


template<class T>
TensorGPU<T>::TensorGPU(int iw, int ih, int im)
{
	std::vector<unsigned> in_dims(3,0);
	in_dims[0] = iw;
	in_dims[1] = ih;
	in_dims[2] = im;
	dims_ = in_dims;
	AllocateMemory();
}

template<class T>
TensorGPU<T>::TensorGPU(const TensorGPU<T>& tens, bool shallow /* = false */) 
{
    if(!shallow){	
        *this = tens;	
    } else {
        data_ = tens.data_;
        dims_ = tens.dims_;
        shallow_ = shallow; //true
    }
}

template<class T>
void TensorGPU<T>::AllocateMemory()
{	
	cutilSafeCall(cudaMalloc((void**)&(data_),sizeof(T)*num_elements()));
	cutilSafeCall(cudaMemset(data_,0,sizeof(T)*num_elements()));
}

template<class T>
void TensorGPU<T>::DeallocateMemory()
{
	if(data_)	{
		//cutilSafeCall(cudaFree((void*)data_));
		//Don't use cutilSafeCall since it can raise exception
		//Usually if cudaFree is not succeed something fatal happened in kernel and next 
		//call of any cuda function will also be unsucsessfull
		cudaFree((void*)data_);
		data_ = NULL;
	}
}

template<class T>
void TensorGPU<T>::ZeroMemory()
{
	cutilSafeCall(cudaMemset(data_,0,sizeof(T)*num_elements()));
}


template<class T>
TensorGPU<T>& TensorGPU<T>::operator =(const TensorGPU<T>& rhs )
{
	if (this == &rhs)      // Same object?
		return *this;
	Destroy();
	dims_ = rhs.dims();
	AllocateMemory();
	cutilSafeCall(cudaMemcpy(data_,rhs.data(),sizeof(T)*num_elements(), cudaMemcpyDeviceToDevice));
	return *this;
}
template<class T>
TensorGPU<T>& TensorGPU<T>::operator =(const Tensor<T>& rhs )
{
//	if (this == &rhs)      // Same object?
//		return *this;
	Destroy();
	dims_  = rhs.dims();
	AllocateMemory();
	cutilSafeCall(cudaMemcpy(data_,rhs.data(),sizeof(T)*num_elements(), cudaMemcpyHostToDevice));
	return *this;
}

} //namespace cudacnn
#endif
