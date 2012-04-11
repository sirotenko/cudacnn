#pragma once

#ifndef _CLAYER_H
#define _CLAYER_H

template <class T, class TF>
class CLayer<Tensor, T, TF> : public CLayerT<Tensor, T, TF>
{
public:
	CLayer(CLayerT<Tensor, T, TF>::Params& params) : CLayerT<Tensor, T, TF>(params){};
	CLayer(UINT inp_width, UINT inp_height, 
		const Tensor<T>& weights, const Tensor<T>& biases, const Tensor<int>& con_map) :
	CLayerT(inp_width, inp_height, weights, biases, con_map) {};

	//Simple implementation (for testing)
	virtual void Propagate(const Tensor<T>& input);
	virtual void BackPropagate(const Tensor<T>& input, Tensor<T>& dedx_prev);
	/* Backpropagate and accumulate Hessian. Before starting Hessian computation iterations,
	it should be reset. See Layer::Reset(). After iterations of Hessian 
	computation it should be averaged before use. See Layer::AverageHessian() */
	virtual void BackPropagateHessian(const Tensor<T>& input, Tensor<T>& d2edx2_prev);
	/* Compute gradient without backpropagating errors */
	virtual void ComputeGradient(const Tensor<T>& input);
	/* Compute Hessian without backpropagating errors */
	virtual void ComputeHessian(const Tensor<T>& input);
	/* Average the Hessian by the number of iterations it */
	virtual void AverageHessian();
	virtual void AdaptWeights(T tau, bool use_hessian, T mu = T(0.0001));
	static void Conv2Valid(const Tensor<T>& input, const Tensor<T>& kernel, Tensor<T>& output);
protected:
	template<bool hessian>
	void BackPropagateTemplate(const Tensor<T>& input, Tensor<T>& dedx_prev);
	template<bool hessian>
	void ComputeDerrivativeTemplate(const Tensor<T>& input);
};

typedef CLayer<Tensor,float, TansigMod<float> > CLayerFTS;
//typedef CLayerT<Tensor,double> CLayerD;

#endif