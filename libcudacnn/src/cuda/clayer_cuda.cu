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

#include "../precomp.hpp"

namespace cudacnn
{

template class CLayer<TensorGPU, float, TansigMod<float> >;
template class CLayer<TensorGPU, float, Tansig<float> >;
template class CLayer<TensorGPU, float, Purelin<float> >;
template class CLayer<TensorGPU, double, TansigMod<double> >;
template class CLayer<TensorGPU, double, Tansig<double> >;
template class CLayer<TensorGPU, double, Purelin<double> >;


#ifdef HAVE_CUDA

template <class T, class TF>
void CLayer<TensorGPU, T, TF>::AverageHessian()
{
	if(num_hessian_accums_)	{
		dim3 threads(min(512,d2e_dw2_.num_elements()),1,1);
		dim3 blocks(iDivUp(d2e_dw2_.num_elements(),512),1,1);
		Average<T><<<blocks, threads>>>(d2e_dw2_, num_hessian_accums_);
		threads = dim3(min(512,d2e_db2_.num_elements()),1,1);
		blocks = dim3(iDivUp(d2e_db2_.num_elements(),512),1,1);
		Average<T><<<blocks, threads>>>(d2e_db2_, num_hessian_accums_);
		num_hessian_accums_ = 0;
	}
}


template <class T, class TF>
void CLayer<TensorGPU, T, TF>::AdaptWeights(T tau, bool use_hessian, T mu)
{
	dim3 threads(MAX_THREADS);
	if(use_hessian){
		dim3 blocks(iDivUp(weights().num_elements(),MAX_THREADS));
		AdaptWeightsKernel<T><<<threads,blocks>>>(weights(), tau, mu, de_dw(), d2e_dw2()); 
		blocks = dim3(iDivUp(biases().num_elements(),MAX_THREADS));
		AdaptWeightsKernel<T><<<threads,blocks>>>(biases(), tau, mu, de_db(), d2e_db2()); 
	}else{
		dim3 blocks(iDivUp(weights().num_elements(),MAX_THREADS));
		AdaptWeightsKernel<T><<<threads,blocks>>>(weights(), tau, de_dw());
		blocks = dim3(iDivUp(biases().num_elements(),MAX_THREADS));
		AdaptWeightsKernel<T><<<threads,blocks>>>(biases(), tau, de_db());
	}
}


//Simple variant. Without extra threads for maximum occupancy
template <class T, class TF, int nthreads>
__global__ void Conv2ValidKernel(const TensorDev<T> inputs, const TensorDev<T> kernels, const TensorDev<T> biases,
								 const TensorDev<int> conn_map, TensorDev<T> outputs)
{
	//__shared__ T smem[nthreads*2];
    T* smem = SharedMemory<T>();
	T* kernels_buf = smem;
	T* sum_buf = smem + nthreads;
	int kx = threadIdx.x;
	int ky = threadIdx.y;
	//int km = threadIdx.z;
	//output coords
	int km = blockIdx.y;
	int y = blockIdx.x / outputs.w();
	int x = blockIdx.x % outputs.w();
	//int tid = threadIdx.z*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	kernels_buf[tid] = 0;
	sum_buf[tid] = 0;
	T out = 0;
	if(kx < kernels.w() && ky < kernels.h()) {
		//Loop for all inputs
		for(int i = 0; i < inputs.d(); ++i) {
			//Load kernel into smem
			kernels_buf[tid] = kernels(kx,ky,i,km);
			__syncthreads();

			sum_buf[tid] = kernels_buf[tid] * inputs(x + kx, y + ky, i);
			__syncthreads();
			volatile T* vsmem = sum_buf;
            SmemReduce<T, nthreads>(vsmem, tid);
			__syncthreads();
			//Check connection
			if(tid == 0){
				out += conn_map(i, km) > 0 ? vsmem[tid] : 0;
			}
		}
	}
	if(tid == 0){
		TF tf;
		outputs(x, y, km) = tf(out + biases[km]);
	}
}

//TODO: remove limitation on 32x32 maximum kernel zise
template <class T, class TF>
void CLayer<TensorGPU, T, TF>::Propagate(const TensorGPU<T>& layer_input )
{
	//TODO: parametrize max threads number
	assert(weights().w() * weights().h() <= MAX_THREADS);
	assert(con_map().w() == layer_input.d());
	assert(con_map().h() == out().d());
	assert(weights().d() == con_map().w());
	assert(weights().d2() == con_map().h());

	dim3 threads(iRoundUpPow2(weights().w()),iRoundUpPow2(weights().h()),1);
	dim3 blocks(out().w()*out().h(),out().d(), 1);
	int nthreads = threads.x*threads.y;
    size_t smem_size = std::max(nthreads*2*sizeof(T), 64*sizeof(T));
	switch(nthreads)
	{
	case 1  : Conv2ValidKernel<T, TF, 1 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	case 2  : Conv2ValidKernel<T, TF, 2 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	case 4  : Conv2ValidKernel<T, TF, 4 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	case 8  : Conv2ValidKernel<T, TF, 8 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	case 16 : Conv2ValidKernel<T, TF, 16 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	case 32 : Conv2ValidKernel<T, TF, 32 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	case 64 : Conv2ValidKernel<T, TF, 64 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	case 128: Conv2ValidKernel<T, TF, 128 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	case 256: Conv2ValidKernel<T, TF, 256 ><<<blocks, threads, smem_size>>>(layer_input, weights(), biases(), 
				con_map(), out_); break;
	default:
		throw std::runtime_error("Incorrect threads number in Propagate");
	}
	cutilCheckMsg("Failed to Propagate in CLayerCuda");
}

template <class T, int nthreads, class TF, bool hessian>
__global__ void BackpropConvKernel(const TensorDev<T> dedx, const TensorDev<T> weights, const TensorDev<T> outs,
							   const TensorDev<int> conn_map, unsigned out_idx, TensorDev<T> de_dx_prev)
{
	T* sum_buf = SharedMemory<T>();
	int kx = threadIdx.x % weights.w();
	int ky = threadIdx.x / weights.w();
	//output coords (output is bigger than input)
	int ix = blockIdx.x % de_dx_prev.w();
	int iy = blockIdx.x / de_dx_prev.w();
	int im = blockIdx.y;

	int kw = weights.w();
	int kh = weights.h();

	int y = iy - ky;
	int x = ix - kx;
	
	if(conn_map(im, out_idx) == 0) return;

	int tid = threadIdx.x;
	sum_buf[tid] = 0;
    __syncthreads();
	if(kx <  kw		  && ky <  kh &&
	    x >= 0		  && y  >= 0  && 
		x <  outs.w() && y  <  outs.h()) {
		//Load kernel into smem
		TF tf;		
		T dedy = hessian ? Sqr(tf.dydx(outs(x,y,out_idx)))*dedx(x, y, out_idx): 
								tf.dydx(outs(x,y,out_idx))*dedx(x, y, out_idx);
		sum_buf[tid] = hessian ?  dedy * Sqr(weights(kx, ky,im, out_idx)) : dedy * weights(kx, ky,im, out_idx);
	}
	__syncthreads();
	volatile T* vsmem = sum_buf;
	SmemReduce<T, nthreads>(vsmem, tid);
	__syncthreads();

	if(tid == 0){
		de_dx_prev(ix, iy, im) += vsmem[tid];
	}

}

template <class T, class TF>
template <bool hessian>
void CLayer<TensorGPU, T, TF>::BackpropagateKernelProxy(const TensorGPU<T>& input, TensorGPU<T>& de_dx_prev)
{	
	assert(weights().w() * weights().h() <= MAX_THREADS);
	assert(con_map().w() * con_map().h() == input.d()*weights().d2());
    assert(de_dx_.HaveSameSize(out_));
    assert(de_dx_prev.HaveSameSize(input));
    

	const TensorGPU<T>& de_dx_t = hessian ? d2e_dx2() : de_dx();

	dim3 threads(iRoundUpPow2(weights().w()*weights().h()),1,1);
	dim3 blocks(input.w()*input.h(),input.d(), 1);
	int nthreads = threads.x;
    //Mimimum size of smem should be at least 64*sizeof(T) to avoid extra checks in reduction kernel and therefore warp divergence
    size_t smem_size = std::max(nthreads*sizeof(T), 64*sizeof(T));

	for(unsigned out_idx = 0; out_idx < out().d(); ++out_idx){
		switch(nthreads)
	 	{
		case 1  : BackpropConvKernel<T, 1 ,   TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		case 2  : BackpropConvKernel<T, 2 ,   TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		case 4  : BackpropConvKernel<T, 4 ,   TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		case 8  : BackpropConvKernel<T, 8 ,   TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		case 16 : BackpropConvKernel<T, 16 ,  TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		case 32 : BackpropConvKernel<T, 32 ,  TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		case 64 : BackpropConvKernel<T, 64 ,  TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		case 128: BackpropConvKernel<T, 128 , TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		case 256: BackpropConvKernel<T, 256 , TF, hessian><<<blocks, threads, smem_size>>>(de_dx_t, weights(), out(), con_map(), out_idx, de_dx_prev); break;
		default:
			throw std::runtime_error("Incorrect threads number in Propagate");
		}
		cutilCheckMsg("Failed to Backpropagate in CLayerCuda");
	}
}


template <class T, int nthreads, class TF, bool hessian>
__global__ void ComputeGradientKernel(const TensorDev<T> dedx, const TensorDev<T> weights, const TensorDev<T> outs,
									  const TensorDev<int> conn_map, const TensorDev<T> inps, TensorDev<T> de_dw, 
									  TensorDev<T> de_db)
{
	T *smem = SharedMemory<T>();

    //Use many threads of 1 block to process several outputs for increasing occupancy
	T* sum_buf = smem + threadIdx.y*(nthreads + outs.w()*outs.h());
	T* dedy_buf = sum_buf + nthreads;
//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200 
//	int im = blockIdx.y;
//	int om = blockIdx.z*blockDim.y + threadIdx.y;
//#else
	int im = blockIdx.y % conn_map.w();
	int om = (blockIdx.y / conn_map.w())*blockDim.y + threadIdx.y;
//#endif
	int kx = blockIdx.x % weights.w();
	int ky = blockIdx.x / weights.w();

	int tid = threadIdx.x;

	int out_size = outs.w() * outs.h();
    //cuassert(im < conn_map.w());
    //cuassert(om < conn_map.h());

	//Compute dedy and put into smem buffer
	for(int out_idx = 0; out_idx < out_size; out_idx += nthreads){
		if(tid + out_idx < out_size){
			int ox = (tid + out_idx) % outs.w();
			int oy = (tid + out_idx) / outs.w();

			TF tf;		
			T dedy = hessian ? Sqr(tf.dydx(outs(ox,oy,om)))*dedx(ox, oy, om): 
				tf.dydx(outs(ox,oy,om))*dedx(ox, oy, om);
			dedy_buf[tid + out_idx] = dedy;
		}
	}
	__syncthreads();
	sum_buf[tid] = 0;
    if(conn_map(im, om) != 0) {
        //Loop for all outputs
		//Prepare dedy * input for reduction
		for(int out_idx = 0; out_idx < out_size; out_idx += nthreads){
			if(tid + out_idx < out_size){
				int ox = (tid + out_idx) % outs.w();
				int oy = (tid + out_idx) / outs.w();

				T inp = hessian ? Sqr(inps(ox + kx, oy + ky, im)) : inps(ox + kx, oy + ky, im);
				sum_buf[tid] += dedy_buf[tid + out_idx] * inp;
			}
		}
		__syncthreads();	
		volatile T* vsmem = sum_buf;
		SmemReduce<T, nthreads>(vsmem, tid);
		__syncthreads();

		if(tid == 0){
			de_dw(kx, ky, im, om) = vsmem[tid];
		}
	}

	//Now compute biases gradient
	if(im == 0){
		sum_buf[tid] = 0;
		for(int out_idx = 0; out_idx < out_size; out_idx += nthreads){
			if(tid + out_idx < out_size){
				int ox = (tid + out_idx) % outs.w();
				int oy = (tid + out_idx) / outs.w();
				sum_buf[tid] += dedy_buf[tid + out_idx];
			}
		}
		__syncthreads();	
		volatile T* vsmem = sum_buf;
		SmemReduce<T, nthreads>(vsmem, tid);
		__syncthreads();

		if(tid == 0){
			de_db[om] = vsmem[tid];
		}
	}

}

template <class T, class TF>
template <bool hessian>
void CLayer<TensorGPU, T, TF>::ComputeGradientKernelProxy(const TensorGPU<T>& input)
{
	assert(con_map().w() * con_map().h() == input.d()*weights().d2());

	const TensorGPU<T>& de_dw_in = hessian ? d2e_dw2() : de_dw();
	const TensorGPU<T>& de_db_in = hessian ? d2e_db2() : de_db();
	const TensorGPU<T>& de_dx_in = hessian ? d2e_dx2() : de_dx();
	//In case when de_dx_ 2d size is greather than max number of threads, loop inside the kernel
	//TODO: CC <1.2 have less than 512 maximum threads. Find some solution better than MAX_THREADS/2
    int nthreads_per_out = min(iRoundUpPow2(de_dx_.w()*de_dx_.h()), MAX_THREADS/2);
    //Try to inrease occupancy by processing several outputs in one block
    int nouts_per_block = max(MAX_OCCUP_THREADS/nthreads_per_out , 1);
    //Restrict the number of outputs
    nouts_per_block = min(nouts_per_block, de_dx_.d());
	dim3 threads(nthreads_per_out,nouts_per_block,1);
    //Use 3rd dimension if CC>=2.0
//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
//    dim3 blocks(weights().w()*weights().h(),input.d(),de_dx_.d()/nouts_per_block);
//#else
	dim3 blocks(weights().w()*weights().h(),input.d()*(de_dx_.d()/nouts_per_block), 1);
//#endif
	int nthreads = threads.x;
	int smem_size = (nthreads + out().w()*out().h())*nouts_per_block*sizeof(T);
	switch(nthreads)
	{
	case 1  : ComputeGradientKernel<T, 1 ,   TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 2  : ComputeGradientKernel<T, 2 ,   TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 4  : ComputeGradientKernel<T, 4 ,   TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 8  : ComputeGradientKernel<T, 8 ,   TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 16 : ComputeGradientKernel<T, 16 ,  TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 32 : ComputeGradientKernel<T, 32 ,  TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 64 : ComputeGradientKernel<T, 64 ,  TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 128: ComputeGradientKernel<T, 128 , TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 256: ComputeGradientKernel<T, 256 , TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	case 512: ComputeGradientKernel<T, 512,  TF, hessian><<<blocks, threads, smem_size>>>(de_dx_in, weights(), out(), con_map(), 
				  input, de_dw_in, de_db_in); break;
	default:
		throw std::runtime_error("Incorrect threads number in ComputeGradientKernelProxy");
	}
	cutilCheckMsg("Failed to Backpropagate in CLayerCuda");

}


template <class T, class TF>
void CLayer<TensorGPU, T, TF>::BackPropagate(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev )
{
	assert(de_dw_.HaveSameSize(weights()));
	assert(de_db_.HaveSameSize(biases()));
	assert(de_dx_.HaveSameSize(out()));
	dedx_prev.ZeroMemory();
	BackpropagateKernelProxy<false>(input, dedx_prev);
	ComputeGradient(input);
}

template <class T, class TF>
void CLayer<TensorGPU, T, TF>::BackPropagateHessian(const TensorGPU<T>& input, TensorGPU<T>& d2edx2_prev )
{
	assert(d2e_dw2_.HaveSameSize(weights()));
	assert(d2e_db2_.HaveSameSize(biases()));
	assert(d2e_dx2_.HaveSameSize(out()));
	d2edx2_prev.ZeroMemory();
	BackpropagateKernelProxy<true>(input, d2edx2_prev);
	ComputeHessian(input);
}

/* Compute gradient without backpropagating errors */
template <class T, class TF>
void CLayer<TensorGPU, T, TF>::ComputeGradient(const TensorGPU<T>& input)
{
	ComputeGradientKernelProxy<false>(input);
}
/* Compute Hessian without backpropagating errors */
template <class T, class TF>
void CLayer<TensorGPU, T, TF>::ComputeHessian(const TensorGPU<T>& input)
{
	ComputeGradientKernelProxy<true>(input);
	num_hessian_accums_++;
}


#endif //HAVE_CUDA
}