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

#ifndef __UTILS_CUH_
#define __UTILS_CUH_

//TODO: make this dependent from compute capability
#define MAX_THREADS 512
//Optimal number of threads for the best occupancy.
//This is kind of heuristic. Good setting for many modern nVidia GeForce cards
//and for not too sophisticated kernels.
#define MAX_OCCUP_THREADS 192

namespace cudacnn
{

inline int iRoundUpPow2(int v){
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
#ifdef __CUDACC__
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
__global__ void Average(cudacnn::TensorDev<T> data, UINT divider)
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < data.num_elements())
		data[x] /= T(divider);
}

template<class T>
__global__ void AdaptWeightsKernel(cudacnn::TensorDev<T> weights, T tau, cudacnn::TensorDev<T> de_dw) 
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < weights.num_elements()){
		weights[x] -= tau*de_dw[x];
	}
}

template<class T>
__global__ void AdaptWeightsKernel(cudacnn::TensorDev<T> weights, T tau, T mu, cudacnn::TensorDev<T> de_dw, cudacnn::TensorDev<T> d2e_dw2) 
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < weights.num_elements()){
		weights[x] -= tau*de_dw[x]/(d2e_dw2[x] + mu);
	}
}

template<class T, class TF>
__global__ void ApplyTransferFunction(const cudacnn::TensorDev<T> input, cudacnn::TensorDev<T> output) 
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < output.num_elements()){
		TF transfer_fnc;
		output[x] = transfer_fnc(input[x]);
	}
}


template<class T, class TF, bool hessian>
__global__ void ApplyTransferFunctionDerriv(const cudacnn::TensorDev<T> fn_output, 
											const cudacnn::TensorDev<T> dedx, cudacnn::TensorDev<T> output) 
{
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	if(x < dedx.num_elements()){
		TF transfer_fnc;
		output[x] = hessian ? Sqr(transfer_fnc.dydx(fn_output[x]))*dedx[x] : 
			transfer_fnc.dydx(fn_output[x])*dedx[x];
	}
}

#endif //__CUDACC__
}


#endif