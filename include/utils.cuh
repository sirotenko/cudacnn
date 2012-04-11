#ifndef __UTILS_CUH_
#define __UTILS_CUH_


inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

//!!!To avoid warp divergence shared memory size should be at least 64*sizeof(T)
template<class T, int nthreads>
__device__ void SmemReduce(volatile T* vsmem, int tid)
{
	if(nthreads >= 512) { if(tid < 256) vsmem[tid] += vsmem[tid+256]; __syncthreads();}
	if(nthreads >= 256) { if(tid < 128) vsmem[tid] += vsmem[tid+128]; __syncthreads();}
	if(nthreads >= 128) { if(tid < 64) vsmem[tid] += vsmem[tid+64];  __syncthreads(); }
	//All these run in a single warp
	if(tid < 32) {
		if(nthreads >= 64) vsmem[tid] += vsmem[tid + 32];
		if(nthreads >= 32) vsmem[tid] += vsmem[tid + 16];
		if(nthreads >= 16) vsmem[tid] += vsmem[tid + 8]; 
		if(nthreads >= 8)  vsmem[tid] += vsmem[tid + 4]; 
		if(nthreads >= 4)  vsmem[tid] += vsmem[tid + 2]; 
		if(nthreads >= 2)  vsmem[tid] += vsmem[tid + 1]; 
	}
}


template<class T>
__global__ void Average(TensorDev<T> data, UINT divider)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < data.num_elements())
		data[x] /= T(divider);
}

template<class T>
__global__ void AdaptWeightsKernel(TensorDev<T> weights, T tau, TensorDev<T> de_dw) 
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < weights.num_elements()){
		weights[x] -= tau*de_dw[x];
	}
}

template<class T>
__global__ void AdaptWeightsKernel(TensorDev<T> weights, T tau, T mu, TensorDev<T> de_dw, TensorDev<T> d2e_dw2) 
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < weights.num_elements()){
		weights[x] -= tau*de_dw[x]/(d2e_dw2[x] + mu);
	}
}

template<class T, class TF>
__global__ void ApplyTransferFunction(const TensorDev<T> input, TensorDev<T> output) 
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < output.num_elements()){
		TF transfer_fnc;
		output[x] = transfer_fnc(input[x]);
	}
}


template<class T, class TF, bool hessian>
__global__ void ApplyTransferFunctionDerriv(const TensorDev<T> fn_output, 
											const TensorDev<T> dedx, TensorDev<T> output) 
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < dedx.num_elements()){
		TF transfer_fnc;
		output[x] = hessian ? Sqr(transfer_fnc.dydx(fn_output[x]))*dedx[x] : 
			transfer_fnc.dydx(fn_output[x])*dedx[x];
	}
}


#endif