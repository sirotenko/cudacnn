#pragma once

#ifndef _CLAYER_CUDA_H
#define _CLAYER_CUDA_H

template <class T, class TF>
class CLayer<TensorGPU, T, TF> : public CLayerT<TensorGPU, T, TF>
{
public:
	CLayer(typename CLayerT<TensorGPU, T, TF>::Params& params) : CLayerT<TensorGPU, T, TF>(params){};
	CLayer(UINT inp_width, UINT inp_height, 
		const TensorGPU<T>& weights, const TensorGPU<T>& biases, const TensorGPU<int>& con_map) :
    CLayerT<TensorGPU, T, TF>(inp_width, inp_height, weights, biases, con_map) {};

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

typedef CLayer<TensorGPU, float, TansigMod<float> > CLayerCudaFTS;
//typedef CLayer<TensorGPU, double> CLayerCudaD;

#endif