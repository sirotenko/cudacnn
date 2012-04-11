#pragma once

#ifndef _PLAYER_CUDA_H
#define _PLAYER_CUDA_H

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

typedef PoolingLayer<TensorGPU,float> SLayerCudaF;
typedef PoolingLayer<TensorGPU,double> SLayerCudaD;


#endif