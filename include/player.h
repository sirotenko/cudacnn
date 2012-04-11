#pragma once
#ifndef _PLAYER_H
#define _PLAYER_H

template <class T>
class PoolingLayer<Tensor, T>: public PoolingLayerT<Tensor, T>
{
public:
	PoolingLayer(const PoolingLayerT<Tensor, T>::Params& params) : PoolingLayerT<Tensor, T>(params) {};
	virtual void Propagate(const Tensor<T>& input);
	virtual void BackPropagate(const Tensor<T>& input, Tensor<T>& dedx_prev);
	/* Backpropagate and accumulate Hessian. Before starting Hessian computation iterations,
	it should be reset. See Layer::Reset(). After iterations of Hessian 
	computation it should be averaged before use. See Layer::AverageHessian() */
	virtual void BackPropagateHessian(const Tensor<T>& input, Tensor<T>& d2edx2_prev);
	/* Compute gradient without backpropagating errors */
	virtual void ComputeGradient(const Tensor<T>& input) {}; //No weights, no gradient
	/* Compute Hessian without backpropagating errors */
	virtual void ComputeHessian(const Tensor<T>& input) {}; //No weights, no hessian
	/* Average the Hessian by the number of iterations it */
    virtual void AverageHessian() {}; //No weights, nothing to average
    virtual void AdaptWeights(T tau, bool use_hessian, T mu = T(0.0001)) {}; //No weights, no adapdation
protected:
	template<bool hessian>
	void BackPropagateTemplateAverage(const Tensor<T>& input, Tensor<T>& dedx_prev);
    template<bool hessian>
    void BackPropagateTemplateMax(const Tensor<T>& input, Tensor<T>& dedx_prev);

    void PropagateAverage(const Tensor<T>& input);
    void PropagateMax(const Tensor<T>& input);

};

typedef PoolingLayer<Tensor, float> PLayerF;
typedef PoolingLayer<Tensor, double> PLayerD;


#endif