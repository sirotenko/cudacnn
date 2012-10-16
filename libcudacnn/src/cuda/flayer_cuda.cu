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

#include <stdexcept>
#include <cublas_v2.h>

#include "../precomp.hpp"

namespace cudacnn
{

template class FLayer<TensorGPU, float, TansigMod<float> >;
template class FLayer<TensorGPU, float, Tansig<float> >;
template class FLayer<TensorGPU, float, Purelin<float> >;
template class FLayer<TensorGPU, double, TansigMod<double> >;
template class FLayer<TensorGPU, double, Tansig<double> >;
template class FLayer<TensorGPU, double, Purelin<double> >;

#ifdef HAVE_CUDA

template<class T>
void ApplyWeightsTemplate(const cublasHandle_t& handle, const TensorGPU<T>& layer_input, const TensorGPU<T>& weights, 
						  const TensorGPU<T>& biases, TensorGPU<T>& out );
template<>
void ApplyWeightsTemplate<float>(const cublasHandle_t& handle, const TensorGPU<float>& layer_input, const TensorGPU<float>& weights, 
						  const TensorGPU<float>& biases, TensorGPU<float>& out )
{	
	//Copy biases to out because of sgemm() syntax uses C as a bias and the output
	cutilSafeCall(cudaMemcpy(out.data(),biases.data(),sizeof(float)*biases.num_elements(), cudaMemcpyDeviceToDevice));

	//Flatten the input
	const float alpha = 1.0;
	const float beta = 1.0;	
	cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
		1,weights.w(), layer_input.num_elements(), 
		&alpha, 
		layer_input.data(), layer_input.num_elements(), 
		weights.data(), weights.w(), 
		&beta, 
		out.data(), out.h());
	cublasCheckMsg(ret, "cublas Sgemm returned an error!\n");
}

template<>
void ApplyWeightsTemplate<double>(const cublasHandle_t& handle, const TensorGPU<double>& layer_input, const TensorGPU<double>& weights, 
						  const TensorGPU<double>& biases, TensorGPU<double>& out )
{
	//Copy biases to out because of sgemm() syntax uses C as a bias and the output
	cutilSafeCall(cudaMemcpy(out.data(),biases.data(),sizeof(double)*biases.num_elements(), cudaMemcpyDeviceToDevice));

	//Flatten the input

	const double alpha = 1.0;
	const double beta = 1.0;	
	cublasStatus_t ret = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
		1,weights.w(), layer_input.num_elements(), 
		&alpha, 
		layer_input.data(), layer_input.num_elements(), 
		weights.data(), weights.w(), 
		&beta, 
		out.data(), out.h());
	cublasCheckMsg(ret, "cublas Sgemm returned an error!\n");
}


template<class T, class TF>
void FLayer<TensorGPU,T, TF>::Propagate(const TensorGPU<T>& layer_input )
{
    //Flatten the output of previous layer
    TensorGPU<T> flat_input(layer_input, true);
    flat_input.Flatten();

	assert(flat_input.num_elements() == this->weights().h());

	ApplyWeightsTemplate<T>(cublas_handle_, flat_input, this->weights(), 
                this->biases(), this->out_);
	dim3 blocks(iDivUp(this->out().num_elements(),MAX_THREADS),1,1); 
	dim3 threads(MAX_THREADS,1,1);
	ApplyTransferFunction<T,TF><<<blocks, threads>>>(this->out_, this->out_);
}



template <class T, int nthreads, bool hessian>
__global__ void FLayerBackpropagateKernel(TensorDev<T> dedy, TensorDev<T> input, TensorDev<T> weights, 
										  TensorDev<T> de_dw, TensorDev<T> de_db, TensorDev<T> de_dx_prev)
{
	//volatile __shared__ T smem[nthreads];
    T* smem = SharedMemory<T>();
	int tx = threadIdx.x;
	int by = blockIdx.y;
	int x = tx;
	int y = by;
	int tid = tx;
	smem[tid] = 0;
	if(x < dedy.num_elements()){
		smem[tid] = hessian ? dedy[x]*Sqr(weights(x,y)) : dedy[x]*weights(x,y);
		//Gradients
		de_dw(x,y) = hessian ? de_dw(x,y) + dedy[x]*Sqr(input[y]): dedy[x]*input[y];
		if(y == 0)
			de_db[x] = hessian ? de_db[x] + dedy[x] : dedy[x];
	}
    volatile T* vsmem = smem;
    SmemReduce<T, nthreads>(vsmem, tid);
	__syncthreads();
	//Copy to destination
	if(x==0) de_dx_prev[y] = smem[0];
	__syncthreads();
}


//Only compute derrivative without backpropagation
template <class T, bool hessian>
__global__ void FLayerComputeDerrivKernel(TensorDev<T> dedy, TensorDev<T> input, 
										  TensorDev<T> de_dw, TensorDev<T> de_db)
{
	int tx = threadIdx.x;
	int by = blockIdx.y;
	int x = tx;
	int y = by;
	if(x < dedy.num_elements()){
		//Gradients
		de_dw(x,y) = hessian ? de_dw(x,y) + dedy[x]*Sqr(input[y]): dedy[x]*input[y];
		if(y == 0)
			de_db[x] = hessian ? de_db[x] + dedy[x] : dedy[x];
	}
}


template <class T, class TF>
template <bool hessian>
inline void FLayer<TensorGPU,T, TF>::BackpropagateKernelProxy(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev)
{
	const TensorGPU<T>& de_dw_in = hessian ? this->d2e_dw2() : this->de_dw();
	const TensorGPU<T>& de_db_in = hessian ? this->d2e_db2() : this->de_db();
	const TensorGPU<T>& de_dx_in = hessian ? this->d2e_dx2() : this->de_dx();

	dim3 threads(MAX_THREADS);
	dim3 blocks(iDivUp(de_dx_in.num_elements(), MAX_THREADS)); 
	TensorGPU<T> dedy(de_dx_in);
	ApplyTransferFunctionDerriv<T, TF, hessian><<<blocks, threads>>>(this->out(), de_dx_in, dedy);
	//TODO: fix this. Weights width can be greather than 1024
        int nthreads = iRoundUpPow2(this->weights().w());
	threads = dim3(nthreads,1,1);
	blocks = dim3(1, this->weights().h(),1); 
        size_t smem_size = std::max(nthreads*sizeof(T), 64*sizeof(T));
	switch(threads.x)
	{
	case 1  : FLayerBackpropagateKernel<T, 1 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	case 2  : FLayerBackpropagateKernel<T,2 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	case 4  : FLayerBackpropagateKernel<T,4 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	case 8  : FLayerBackpropagateKernel<T,8 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	case 16 : FLayerBackpropagateKernel<T,16 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	case 32 : FLayerBackpropagateKernel<T,32 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	case 64 : FLayerBackpropagateKernel<T,64 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	case 128: FLayerBackpropagateKernel<T,128 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	case 256: FLayerBackpropagateKernel<T,256 , hessian><<<blocks, threads, smem_size>>>(dedy, input, 
				this->weights(), de_dw_in, de_db_in, dedx_prev); break;
	default:
		throw std::runtime_error("Incorrect threads number in BackpropagateKernelProxy");
	}

	cutilCheckMsg("Failed to apply transfer function in FLayer");
}


template <class T, class TF>
void FLayer<TensorGPU,T, TF>::BackPropagate(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev)
{
    //Flatten the output of previous layer
    TensorGPU<T> flat_input(input, true);
    flat_input.Flatten();
    TensorGPU<T> flat_dedx_prev(dedx_prev, true);
    flat_dedx_prev.Flatten();

	assert(flat_input.num_elements() == this->weights().h());
	assert(de_dw_.HaveSameSize(this->weights()));
	assert(de_db_.HaveSameSize(this->biases()));
	//Require only the same number of elements, since last CLayer usually flattened
	assert(flat_dedx_prev.num_elements() == flat_input.num_elements());

	//TODO: Remove the limitation
	assert(weights().w() <= MAX_THREADS);
	BackpropagateKernelProxy<false>(flat_input, flat_dedx_prev);
}

template <class T, class TF>
void FLayer<TensorGPU,T, TF>::BackPropagateHessian(const TensorGPU<T>& input, TensorGPU<T>& d2edx2_prev)
{
    //Flatten the output of previous layer
    TensorGPU<T> flat_input(input, true);
    flat_input.Flatten();
    TensorGPU<T> flat_d2edx2_prev(d2edx2_prev, true);
    flat_d2edx2_prev.Flatten();

	assert(flat_input.num_elements() == this->weights().h());
	assert(this->d2e_dw2_.HaveSameSize(this->weights()));
	assert(this->d2e_db2_.HaveSameSize(this->biases()));
	assert(flat_d2edx2_prev.num_elements() == flat_input.num_elements());
	//Assume this for now. If the weights matrix is bigger than use cublas method
	assert(this->weights().w() <= MAX_THREADS);
	BackpropagateKernelProxy<true>(flat_input, flat_d2edx2_prev);
	//cutilSafeCall(cudaThreadSynchronize());
	this->num_hessian_accums_++;
}

template <class T, class TF>
void FLayer<TensorGPU,T, TF>::AverageHessian()
{
	if(this->num_hessian_accums_)	{
		dim3 threads(min(512,this->d2e_dw2_.num_elements()),1,1);
		dim3 blocks(iDivUp(this->d2e_dw2_.num_elements(),512));
		Average<T><<<blocks, threads>>>(this->d2e_dw2_, this->num_hessian_accums_);
		threads = dim3(min(512,this->d2e_db2_.num_elements()),1,1);
		blocks = dim3(iDivUp(this->d2e_db2_.num_elements(),512));
		Average<T><<<blocks, threads>>>(this->d2e_db2_, this->num_hessian_accums_);
		this->num_hessian_accums_ = 0;
	}
}

template <class T, class TF>
void FLayer<TensorGPU,T, TF>::AdaptWeights(T tau, bool use_hessian, T mu)
{
	dim3 threads(MAX_THREADS);
	if(use_hessian){
		dim3 blocks(iDivUp(this->weights().num_elements(),MAX_THREADS));
		AdaptWeightsKernel<T><<<threads,blocks>>>(this->weights(), tau, mu, this->de_dw(), this->d2e_dw2()); 
		blocks = dim3(iDivUp(this->biases().num_elements(),MAX_THREADS));
		AdaptWeightsKernel<T><<<threads,blocks>>>(this->biases(), tau, mu, this->de_db(), this->d2e_db2()); 
	}else{
		dim3 blocks(iDivUp(this->weights().num_elements(),MAX_THREADS));
		AdaptWeightsKernel<T><<<threads,blocks>>>(this->weights(), tau, this->de_dw());
		blocks = dim3(iDivUp(this->biases().num_elements(),MAX_THREADS));
		AdaptWeightsKernel<T><<<threads,blocks>>>(this->biases(), tau, this->de_db());
	}
}

template <class T, class TF>
template <bool hessian>
void FLayer<TensorGPU, T, TF>::ComputeGradientKernelProxy(const TensorGPU<T>& input)
{
	const TensorGPU<T>& de_dw_in = hessian ? this->d2e_dw2() : this->de_dw();
	const TensorGPU<T>& de_db_in = hessian ? this->d2e_db2() : this->de_db();
	const TensorGPU<T>& de_dx_in = hessian ? this->d2e_dx2() : this->de_dx();

	dim3 threads(MAX_THREADS);
	dim3 blocks(iDivUp(de_dx_in.num_elements(), MAX_THREADS)); 
	TensorGPU<T> dedy(de_dx_in);
	ApplyTransferFunctionDerriv<T, TF, hessian><<<blocks, threads>>>(this->out(), de_dx_in, dedy);
	//TODO: fix this. Weights width can be greather than 1024
	threads = dim3(iRoundUpPow2(this->weights().w()),1,1);
	blocks = dim3(1, this->weights().h(),1); 
	FLayerComputeDerrivKernel<T, hessian><<<blocks, threads>>>(dedy, input, de_dw_in, de_db_in);
	cutilCheckMsg("Failed to compute derrivative in FLayer");
}

/* Compute gradient without backpropagating errors */
template <class T, class TF>
void FLayer<TensorGPU, T, TF>::ComputeGradient(const TensorGPU<T>& input)
{
    //Flatten the output of previous layer
    TensorGPU<T> flat_input(input, true);
    flat_input.Flatten();

	ComputeGradientKernelProxy<false>(flat_input);
}
/* Compute Hessian without backpropagating errors */
template <class T, class TF>
void FLayer<TensorGPU, T, TF>::ComputeHessian(const TensorGPU<T>& input)
{
    //Flatten the output of previous layer
    TensorGPU<T> flat_input(input, true);
    flat_input.Flatten();
	ComputeGradientKernelProxy<true>(flat_input);
	this->num_hessian_accums_++;
}

#endif //HAVE_CUDA
}