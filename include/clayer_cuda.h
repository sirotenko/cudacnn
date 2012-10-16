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

#ifndef _CLAYER_CUDA_H
#define _CLAYER_CUDA_H

namespace cudacnn
{

//Throws an exception, if compiled without CUDA support
#ifdef HAVE_CUDA

template <class T, class TF>
class CLayer<TensorGPU, T, TF> : public CLayerT<TensorGPU, T, TF>
{
public:
	CLayer(typename CLayerT<TensorGPU, T, TF>::Params& params) : CLayerT<TensorGPU, T, TF>(params){};
	CLayer(UINT inp_width, UINT inp_height, bool is_trainable,
		const TensorGPU<T>& weights, const TensorGPU<T>& biases, const TensorGPU<int>& con_map) :
    CLayerT<TensorGPU, T, TF>(inp_width, inp_height, is_trainable, weights, biases, con_map) {};

	//virtual int Destroy();
	virtual void Propagate(const TensorGPU<T>& input);
	virtual void BackPropagate(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev);
	/* Backpropagate and accumulate Hessian. Before starting Hessian computation iterations,
	it should be reset. See Layer::Reset(). After iterations of Hessian 
	computation it should be averaged before use. See Layer::AverageHessian() */
	virtual void BackPropagateHessian(const TensorGPU<T>& input, TensorGPU<T>& d2edx2_prev);
	/* Compute gradient without backpropagating errors */
	virtual void ComputeGradient(const TensorGPU<T>& input);
	/* Compute Hessian without backpropagating errors */
	virtual void ComputeHessian(const TensorGPU<T>& input);
	/* Average the Hessian by the number of iterations it */
	virtual void AverageHessian();
	/* Apply gradients to weights with learning coefficient tau either using hessian or not */
	virtual void AdaptWeights(T tau, bool use_hessian, T mu = T(0.0001));
private:
	template <bool hessian>
	void BackpropagateKernelProxy(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev);
	template <bool hessian>
	void ComputeGradientKernelProxy(const TensorGPU<T>& input);  
	
};

#else
template <class T, class TF>
class CLayer<TensorGPU, T, TF> : public CLayerT<TensorGPU, T, TF>
{
public:
	CLayer(typename CLayerT<TensorGPU, T, TF>::Params& params) : CLayerT<TensorGPU, T, TF>(params){ std::runtime_error("cudacnn lib compiled without CUDA support");};
	CLayer(UINT inp_width, UINT inp_height, bool is_trainable,
		const TensorGPU<T>& weights, const TensorGPU<T>& tf_params, const TensorGPU<int>& conn_map) :
    CLayerT<TensorGPU, T, TF>(inp_width, inp_height, is_trainable, weights, tf_params, conn_map) {std::runtime_error("cudacnn lib compiled without CUDA support");};

	//virtual int Destroy();
	virtual void Propagate(const TensorGPU<T>& input) {std::runtime_error("cudacnn lib compiled without CUDA support");};
	virtual void BackPropagate(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev) {std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Backpropagate and accumulate Hessian. Before starting Hessian computation iterations,
	it should be reset. See Layer::Reset(). After iterations of Hessian 
	computation it should be averaged before use. See Layer::AverageHessian() */
	virtual void BackPropagateHessian(const TensorGPU<T>& input, TensorGPU<T>& d2edx2_prev) {std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Compute gradient without backpropagating errors */
	virtual void ComputeGradient(const TensorGPU<T>& input) {std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Compute Hessian without backpropagating errors */
	virtual void ComputeHessian(const TensorGPU<T>& input) {std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Average the Hessian by the number of iterations it */
	virtual void AverageHessian() {std::runtime_error("cudacnn lib compiled without CUDA support");};
	/* Apply gradients to weights with learning coefficient tau either using hessian or not */
	virtual void AdaptWeights(T tau, bool use_hessian, T mu = T(0.0001)) {std::runtime_error("cudacnn lib compiled without CUDA support");};

	virtual void ComputeDerivativeUsingOut(const TensorGPU<T>& input, const TensorGPU<T>& out_d) {std::runtime_error("cudacnn lib compiled without CUDA support");};

private:
	template <bool hessian>
	void BackpropagateKernelProxy(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev) {std::runtime_error("cudacnn lib compiled without CUDA support");}
	template <bool hessian>
	void ComputeGradientKernelProxy(const TensorGPU<T>& input) {std::runtime_error("cudacnn lib compiled without CUDA support");}  
	
};

#endif //HAVE_CUDA

typedef CLayer<TensorGPU, float, TansigMod<float> > CLayerCudaFTS;
//typedef CLayer<TensorGPU, double> CLayerCudaD;

}

#endif