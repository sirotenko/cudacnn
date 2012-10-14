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

#ifndef _PLAYER_CUDA_H
#define _PLAYER_CUDA_H

namespace cudacnn
{
//Throws an exception, if compiled without CUDA support
#ifdef HAVE_CUDA

template <class T>
class PoolingLayer<TensorGPU,T>: public PoolingLayerT<TensorGPU, T>
{
public:
	PoolingLayer(const typename PoolingLayerT<TensorGPU, T>::Params& params) : PoolingLayerT<TensorGPU, T>(params) {};
	//virtual int Destroy();
	virtual void Propagate(const TensorGPU<T>& input);
	virtual void BackPropagate(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev);
	/* Backpropagate and accumulate Hessian. Before starting Hessian computation iterations,
	it should be reset. See Layer::Reset(). After iterations of Hessian 
	computation it should be averaged before use. See Layer::AverageHessian() */
	virtual void BackPropagateHessian(const TensorGPU<T>& input, TensorGPU<T>& d2edx2_prev);
	/* Compute gradient without backpropagating errors */
	virtual void ComputeGradient(const TensorGPU<T>& input) {};
	/* Compute Hessian without backpropagating errors */
	virtual void ComputeHessian(const TensorGPU<T>& input) {};
	/* Average the Hessian by the number of iterations it */
    virtual void AverageHessian() {};
    virtual void AdaptWeights(T tau, bool use_hessian, T mu = T(0.0001)) {};
protected:
	template <class TransfFuncType>
	void PropagateKernelProxy(const TensorGPU<T>& layer_input);
	template <bool hessian>
	void BackpropagateKernelProxy(const TensorGPU<T>& dedx, const TensorGPU<T>& input);
	template <class TransfFuncType, bool hessian>
	void BackpropagateKernelProxy(const TensorGPU<T>& dedx, const TensorGPU<T>& input);
};
#else
template <class T>
class PoolingLayer<TensorGPU,T>: public PoolingLayerT<TensorGPU, T>
{
public:
	PoolingLayer(const typename PoolingLayerT<TensorGPU, T>::Params& params) : PoolingLayerT<TensorGPU, T>(params) { std::runtime_error("cudacnn lib compiled without CUDA support");};
	//virtual int Destroy();
	virtual void Propagate(const TensorGPU<T>& input){ std::runtime_error("cudacnn lib compiled without CUDA support");};
	virtual void BackPropagate(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev){ std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Backpropagate and accumulate Hessian. Before starting Hessian computation iterations,
	it should be reset. See Layer::Reset(). After iterations of Hessian 
	computation it should be averaged before use. See Layer::AverageHessian() */
	virtual void BackPropagateHessian(const TensorGPU<T>& input, TensorGPU<T>& d2edx2_prev){ std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Compute gradient without backpropagating errors */
	virtual void ComputeGradient(const TensorGPU<T>& input) { std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Compute Hessian without backpropagating errors */
	virtual void ComputeHessian(const TensorGPU<T>& input) { std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Average the Hessian by the number of iterations it */
    virtual void AverageHessian() {};
    virtual void AdaptWeights(T tau, bool use_hessian, T mu = T(0.0001)) { std::runtime_error("cudacnn lib compiled without CUDA support");};

	virtual void ComputeDerivativeUsingOut(const TensorGPU<T>& input, const TensorGPU<T>& out_d) { std::runtime_error("cudacnn lib compiled without CUDA support");};

protected:
	template <class TransfFuncType>
	void PropagateKernelProxy(const TensorGPU<T>& layer_input) { std::runtime_error("cudacnn lib compiled without CUDA support");};
	template <bool hessian>
	void BackpropagateKernelProxy(const TensorGPU<T>& dedx, const TensorGPU<T>& input) { std::runtime_error("cudacnn lib compiled without CUDA support");};
	template <class TransfFuncType, bool hessian>
	void BackpropagateKernelProxy(const TensorGPU<T>& dedx, const TensorGPU<T>& input) { std::runtime_error("cudacnn lib compiled without CUDA support");};
};
#endif //HAVE_CUDA

typedef PoolingLayer<TensorGPU,float> SLayerCudaF;
typedef PoolingLayer<TensorGPU,double> SLayerCudaD;

}

#endif