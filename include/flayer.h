#pragma once

#ifndef _FLAYER_H
#define _FLAYER_H

template <class T, class TF>
class FLayer<Tensor, T, TF>: public FLayerT<Tensor, T, TF>
{
public:
	FLayer(const Tensor<T>& weights, const Tensor<T>& biases) : FLayerT<Tensor, T, TF>(weights, biases){};
	void Propagate(const Tensor<T>& input);
	void BackPropagate(const Tensor<T>& input, Tensor<T>& dedx_prev);
	/* Backpropagate and accumulate Hessian. Before starting Hessian computation iterations,
	it should be reset. See Layer::Reset(). After iterations of Hessian 
	computation it should be averaged before use. See Layer::AverageHessian() */
	virtual void BackPropagateHessian(const Tensor<T>& input, Tensor<T>& d2edx2_prev);
	/* Compute gradient without backpropagating errors */
	virtual void ComputeGradient(const Tensor<T>& input);
	/* Compute Hessian without backpropagating errors */
	virtual void ComputeHessian(const Tensor<T>& input);
	/* Average the Hessian by the number of iterations it */
	void AverageHessian();
	virtual void AdaptWeights(T tau, bool use_hessian, T mu = T(0.0001));
protected:
	template<bool hessian>
	void ComputeDerrivativeTemplate(const Tensor<T>& input);
};

typedef FLayer<Tensor, float, TansigMod<float> > FLayerFTS;
//typedef FLayer<Tensor, double> FLayerD;


#endif