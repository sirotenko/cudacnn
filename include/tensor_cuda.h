#include "common.h"
#include "tensor.h"
#include "exceptions.h"


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
		cuassert(idx < num_elements());
		return data_[idx];
	}
	__device__ inline T& operator[](int idx) const 
	{
		cuassert(idx < num_elements());
		return data_[idx];
	}
	__device__ inline T& operator() (unsigned x, unsigned y, unsigned n)
	{
		cuassert(num_dims() <= 3);
		cuassert(x < w() && y < h() && n < d());
		return data_[n*w()*h() + x + y*w()];
	}
	__device__ inline T operator() (unsigned x, unsigned y, unsigned n) const
	{
		cuassert(num_dims() <= 3);
		cuassert(x < w() && y < h() && n < d());
		return data_[x + y*w() + n*w()*h()];
	}
	__device__ inline T& operator() (unsigned x, unsigned y)
	{
		cuassert(num_dims() <= 2);
		cuassert(x < w() && y < h());
		return data_[x + y*w()];
	}
	__device__ inline T  operator() (unsigned x, unsigned y) const
	{
		cuassert(num_dims() <= 2);
		cuassert(x < w() && y < h());
		return data_[x + y*w()];
	}
	__device__ inline T& operator() (unsigned x, unsigned y, unsigned m, unsigned n)
	{
		cuassert(num_dims() <= 4);
		cuassert(x < w() && y < h() && m < d() && n < d2());
		return data_[n*d()*w()*h() + m*w()*h() + y*w() + x ];
	}
	__device__ inline T operator() (unsigned x, unsigned y, unsigned m, unsigned n) const
	{
		cuassert(num_dims() <= 4);
		cuassert(x < w() && y < h() && m < d() && n < d2());
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