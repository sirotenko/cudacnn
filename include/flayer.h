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

#ifndef _FLAYER_H
#define _FLAYER_H

namespace cudacnn
{

template <class T, class TF>
class FLayer<Tensor, T, TF>: public FLayerT<Tensor, T, TF>
{
public:
	FLayer(const Tensor<T>& weights, const Tensor<T>& biases, bool is_trainable) : FLayerT<Tensor, T, TF>(weights, biases, is_trainable){};
	FLayer(typename Layer<Tensor,T>::ILoadSaveObject* save_load_obj) : 	FLayerT<Tensor, T, TF>(save_load_obj) {};
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
	virtual ~FLayer() {};
protected:
	template<bool hessian>
	void ComputeDerrivativeTemplate(const Tensor<T>& input);
};

typedef FLayer<Tensor, float, TansigMod<float> > FLayerFTS;
typedef FLayer<Tensor, double, TansigMod<double> > FLayerDTS;

}

#endif