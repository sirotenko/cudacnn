#pragma once

#ifndef _FLAYER_CUDA_H
#define _FLAYER_CUDA_H


template <class T, class TF>
class FLayer<TensorGPU, T, TF>: public FLayerT<TensorGPU, T, TF>
{
public:
	FLayer(const TensorGPU<T>& weights, const TensorGPU<T>& biases) : FLayerT<TensorGPU, T, TF>(weights, biases)
    {
        cublasCheckMsg(cublasCreate(&cublas_handle_), "cublasCreate() error!\n");
    };
    virtual ~FLayer()
    {
        //cublasCheckMsg(cublasDestroy(cublas_handle_), "cublasDestroy() error!\n");
        cublasDestroy(cublas_handle_);
    }

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
	virtual void AdaptWeights(T tau, bool use_hessian, T mu = T(0.0001));
private:
    template <bool hessian>
    void BackpropagateKernelProxy(const TensorGPU<T>& input, TensorGPU<T>& dedx_prev);

	template <bool hessian>
	void ComputeGradientKernelProxy(const TensorGPU<T>& input);
    cublasHandle_t cublas_handle_;
};

typedef FLayer<TensorGPU, float, TansigMod<float> > FLayerCudaFTS;

#endif