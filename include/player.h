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

#ifndef _PLAYER_H
#define _PLAYER_H

namespace cudacnn
{

template <class T>
class PoolingLayer<Tensor, T>: public PoolingLayerT<Tensor, T>
{
public:
	PoolingLayer(const typename PoolingLayerT<Tensor, T>::Params& params) : PoolingLayerT<Tensor, T>(params) {};
	PoolingLayer(typename Layer<Tensor,T>::ILoadSaveObject* save_load_obj) : PoolingLayerT<Tensor, T>(save_load_obj) {};
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

}

#endif