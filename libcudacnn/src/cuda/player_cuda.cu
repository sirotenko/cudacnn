
#include "common.h"
#include <stdexcept>
#include "layer.hpp"
#include "tensor.h"
#include "player_cuda.h"
#include "tensor_cuda.h"
#include "assert.h"
//#include "transfer_functions_cu.h"
#include "utils.cuh"
#include "transfer_functions_cu.h"

//Instantiate
template PoolingLayer<TensorGPU, float>;
template PoolingLayer<TensorGPU, double>;


template <class T, int nthreads>
__global__ void SubsampleKernel(const TensorDev<T> inputs, TensorDev<T> output)
{
	int sx = inputs.w() / output.w();
	int sy = inputs.h() / output.h();
	int tx = threadIdx.x%sx;
	int ty = threadIdx.x/sx;
	int x = blockIdx.x%output.w(); 
	int y = blockIdx.x/output.w();
	int m = blockIdx.y;

	int tid = threadIdx.x;
	__shared__ T smem[nthreads];
	smem[tid] = 0;

	if(tx < sx && ty < sy){
		smem[tid] = inputs(x*sx + tx, y*sy + ty, m);
	}
	volatile T* vsmem = smem;
	////Reduction
	__syncthreads();
	if(nthreads >= 256){
		if(tid < 128) vsmem[tid] += vsmem[tid+128];
		__syncthreads();
	}
	if(nthreads >= 128){
		if(tid < 64) 
			vsmem[tid] += vsmem[tid+64];
		__syncthreads();
	}

	//All these run in a single warp
	if(nthreads >= 64) if(tid < 32) vsmem[tid] += vsmem[tid + 32];
	if(nthreads >= 32) if(tid < 16) vsmem[tid] += vsmem[tid + 16];
	if(nthreads >= 16) if(tid < 8) vsmem[tid] += vsmem[tid + 8];
	if(nthreads >= 8)  if(tid < 4) vsmem[tid] += vsmem[tid + 4];
	if(nthreads >= 4)  if(tid < 2) vsmem[tid] += vsmem[tid + 2];
	if(nthreads >= 2)  if(tid < 1) vsmem[tid] += vsmem[tid + 1];

	if(tid == 0)
		output(x,y,m) = vsmem[0]/(sx*sy);
}
template <class T, int nthreads>
__global__ void MaxPoolingKernel(const TensorDev<T> inputs, TensorDev<T> output)
{
	int sx = inputs.w() / output.w();
	int sy = inputs.h() / output.h();
	int tx = threadIdx.x%sx;
	int ty = threadIdx.x/sx;
	int x = blockIdx.x%output.w(); 
	int y = blockIdx.x/output.w();
	int m = blockIdx.y;

	int tid = threadIdx.x;
	__shared__ T smem[nthreads];
	smem[tid] = 0;

	if(tx < sx && ty < sy){
		smem[tid] = inputs(x*sx + tx, y*sy + ty, m);
	}
	volatile T* vsmem = smem;
	////Reduction
	__syncthreads();
	if(nthreads >= 256){
		if(tid < 128) vsmem[tid] += vsmem[tid+128];
		__syncthreads();
	}
	if(nthreads >= 128){
		if(tid < 64) 
			vsmem[tid] += vsmem[tid+64];
		__syncthreads();
	}

	//All these run in a single warp
	if(nthreads >= 64) if(tid < 32) vsmem[tid] = max(vsmem[tid + 32], vsmem[tid]);
	if(nthreads >= 32) if(tid < 16) vsmem[tid] = max(vsmem[tid + 16], vsmem[tid]);
	if(nthreads >= 16) if(tid < 8) vsmem[tid] = max(vsmem[tid + 8], vsmem[tid]);
	if(nthreads >= 8)  if(tid < 4) vsmem[tid] = max(vsmem[tid + 4], vsmem[tid]);
	if(nthreads >= 4)  if(tid < 2) vsmem[tid] = max(vsmem[tid + 2], vsmem[tid]);
	if(nthreads >= 2)  if(tid < 1) vsmem[tid] = max(vsmem[tid + 1], vsmem[tid]);

	if(tid == 0)
		output(x,y,m) = vsmem[0];
}


template <class T>
void PoolingLayer<TensorGPU, T>::Propagate(const TensorGPU<T>& layer_input )
{
	dim3 blocks(out().w()*out().h(),out().d(),1); 
	int nthreads = iRoundUpPow2(sx_*sy_);
	dim3 threads(nthreads,1,1);
    switch(pooling_type_)
    {
        //Without cast CUDA compiler gives a warning that expression must be an integral type
    case static_cast<int>(eAverage):
	    switch(nthreads)
	    {
	    case 1  : SubsampleKernel<T, 1 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 2  : SubsampleKernel<T, 2 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 4  : SubsampleKernel<T, 4 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 8  : SubsampleKernel<T, 8 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 16 : SubsampleKernel<T, 16 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 32 : SubsampleKernel<T, 32 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 64 : SubsampleKernel<T, 64 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 128: SubsampleKernel<T, 128 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 256: SubsampleKernel<T, 256 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    default:
		    throw std::runtime_error("Incorrect threads number in Propagate");
	    }
        break;
    case static_cast<int>(eMax):
	    switch(nthreads)
	    {
	    case 1  : MaxPoolingKernel<T, 1 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 2  : MaxPoolingKernel<T, 2 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 4  : MaxPoolingKernel<T, 4 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 8  : MaxPoolingKernel<T, 8 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 16 : MaxPoolingKernel<T, 16 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 32 : MaxPoolingKernel<T, 32 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 64 : MaxPoolingKernel<T, 64 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 128: MaxPoolingKernel<T, 128 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    case 256: MaxPoolingKernel<T, 256 ><<<blocks, threads>>>(layer_input,  
				     out_); break;
	    default:
		    throw std::runtime_error("Incorrect threads number in Propagate");
	    }
        break;
    default:
        throw std::runtime_error("Unknown pooling type");

    }
	cutilCheckMsg("Failed to propagate data in player on cuda");
}

template <class T, int nthreads, bool hessian>
__global__ void BakpropagateSubsampleKernel(TensorDev<T> dedx, TensorDev<T> de_dx_prev)
{
	int sx = de_dx_prev.w() / dedx.w();
	int sy = de_dx_prev.h() / dedx.h();
	int tx = threadIdx.x%sx;
	int ty = threadIdx.x/sx;
	int x = blockIdx.x%dedx.w(); 
	int y = blockIdx.x/dedx.w();
	int m = blockIdx.y;

	if(tx < sx && ty < sy){
		de_dx_prev(x*sx + tx, y*sy + ty, m) = dedx(x,y,m)/(sx*sy);
	}
}
template <class T, int nthreads, bool hessian>
__global__ void BakpropagateMaxPoolingKernel(TensorDev<T> input, TensorDev<T> output,
                                             TensorDev<T> dedx, TensorDev<T> de_dx_prev)
{
	int sx = de_dx_prev.w() / dedx.w();
	int sy = de_dx_prev.h() / dedx.h();
	int tx = threadIdx.x%sx;
	int ty = threadIdx.x/sx;
	int x = blockIdx.x%dedx.w(); 
	int y = blockIdx.x/dedx.w();
	int m = blockIdx.y;

	if(tx < sx && ty < sy){
		de_dx_prev(x*sx + tx, y*sy + ty, m) = 
            input(x*sx + tx, y*sy + ty, m) == output(x,y,m) ? dedx(x,y,m) : 0;
	}
}


template <class T>
template <bool hessian>
void PoolingLayer<TensorGPU, T>::BackpropagateKernelProxy(const TensorGPU<T>& input, const TensorGPU<T>& dedx_prev)
{
	const TensorGPU<T>& de_dx_in = hessian ? d2e_dx2() : de_dx();
	dim3 blocks(out().w()*out().h(),out().d(),1); 
	int nthreads = iRoundUpPow2(sx_*sy_);
	dim3 threads(nthreads,1,1);
    switch(pooling_type_)
    {
    case static_cast<int>(eAverage):
	    switch(nthreads)
	    {
	    case 1  : BakpropagateSubsampleKernel<T, 1 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    case 2  : BakpropagateSubsampleKernel<T, 2 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    case 4  : BakpropagateSubsampleKernel<T, 4 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    case 8  : BakpropagateSubsampleKernel<T, 8 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    case 16 : BakpropagateSubsampleKernel<T, 16 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    case 32 : BakpropagateSubsampleKernel<T, 32 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    case 64 : BakpropagateSubsampleKernel<T, 64 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    case 128: BakpropagateSubsampleKernel<T, 128 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    case 256: BakpropagateSubsampleKernel<T, 256 , hessian><<<blocks, threads>>>(de_dx_in,  
				      dedx_prev); break;
	    default:
		    throw std::runtime_error("Incorrect threads number in Propagate");
	    }
        break;
    case static_cast<int>(eMax):
	    switch(nthreads)
	    {
	    case 1  : BakpropagateMaxPoolingKernel<T, 1 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    case 2  : BakpropagateMaxPoolingKernel<T, 2 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    case 4  : BakpropagateMaxPoolingKernel<T, 4 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    case 8  : BakpropagateMaxPoolingKernel<T, 8 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    case 16 : BakpropagateMaxPoolingKernel<T, 16 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    case 32 : BakpropagateMaxPoolingKernel<T, 32 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    case 64 : BakpropagateMaxPoolingKernel<T, 64 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    case 128: BakpropagateMaxPoolingKernel<T, 128 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    case 256: BakpropagateMaxPoolingKernel<T, 256 , hessian><<<blocks, threads>>>(input, out_, de_dx_in,  
				      dedx_prev); break;
	    default:
		    throw std::runtime_error("Incorrect threads number in Propagate");
	    }
        break;
    default:
        throw std::runtime_error("Unknown pooling type");

    }
}


template <class T>
void PoolingLayer<TensorGPU, T>::BackPropagate(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev)
{
	assert(dedx_prev.HaveSameSize(input));
	BackpropagateKernelProxy<false>(input, dedx_prev);
}

template <class T>
void PoolingLayer<TensorGPU, T>::BackPropagateHessian(const TensorGPU<T>& input, TensorGPU<T>& d2edx2_prev)
{
	assert(d2edx2_prev.HaveSameSize(input));
	BackpropagateKernelProxy<true>(input, d2edx2_prev);
}


