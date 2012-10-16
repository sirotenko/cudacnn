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

#include "precomp.hpp"

namespace cudacnn
{

template class FLayer<Tensor, float, TansigMod<float> >;
template class FLayer<Tensor, float, Tansig<float> >;
template class FLayer<Tensor, float, Purelin<float> >;
template class FLayer<Tensor, double, TansigMod<double> >;
template class FLayer<Tensor, double, Tansig<double> >;
template class FLayer<Tensor, double, Purelin<double> >;

//Simple implementation. Just for testing
template<class T, class TF>
void FLayer<Tensor,T,TF>::Propagate(const Tensor<T>& input)
{
        this->out_.ZeroMemory();
    //Flatten the output of previous layer
    Tensor<T> flat_input(input, true);
    flat_input.Flatten();

	assert(flat_input.num_elements() == this->weights_.h());

	for (UINT n = 0; n < this->weights_.w(); ++n) {	
		for (UINT i = 0; i < flat_input.num_elements(); ++i)	{
			this->out_[n] += flat_input[i]*this->weights_(n,i);
		}
		this->out_[n] = this->transfer_function_(this->out_[n] + this->biases_[n]);
	}
}
template<class T, class TF>
void FLayer<Tensor,T,TF>::BackPropagate(const Tensor<T>& input, Tensor<T>& dedx_prev)
{
    //Flatten the output of previous layer
    Tensor<T> flat_input(input, true);
    flat_input.Flatten();

    //Flatten the dedx of previous layer
    Tensor<T> flat_dedx_prev(dedx_prev, true);
    flat_dedx_prev.Flatten();

	assert(this->de_dx_.num_elements() == this->out().num_elements());
	assert(this->de_dw_.HaveSameSize(this->weights()));
	assert(this->de_db_.HaveSameSize(this->biases()));
	assert(flat_dedx_prev.num_elements() == flat_input.num_elements());
	flat_dedx_prev.ZeroMemory();
	for (UINT no = 0; no < this->out().num_elements(); ++no) {
		T dedy = this->transfer_function_.dydx(this->out()[no])*this->de_dx_[no];
		for (UINT ni = 0; ni < this->weights().h(); ++ni) {
			this->de_dw_(no,ni) = dedy*flat_input[ni];
			flat_dedx_prev[ni] += this->weights_(no,ni)*dedy;
		}
		this->de_db_[no] = dedy;//[no];
	}
}

template<class T, class TF>
void FLayer<Tensor,T,TF>::BackPropagateHessian(const Tensor<T>& input, Tensor<T>& d2edx2_prev)
{
    //Flatten the output of previous layer
    Tensor<T> flat_input(input, true);
    flat_input.Flatten();
    //Flatten the dedx of previous layer
    Tensor<T> flat_d2edx2_prev(d2edx2_prev, true);
    flat_d2edx2_prev.Flatten();

	assert(this->d2e_dx2_.num_elements() == this->out().num_elements());
	assert(this->d2e_dw2_.HaveSameSize(this->weights()));
	assert(this->d2e_db2_.HaveSameSize(this->biases()));
	assert(flat_d2edx2_prev.num_elements() == flat_input.num_elements());
	flat_d2edx2_prev.ZeroMemory();
	for (UINT no = 0; no < this->out().num_elements(); ++no) {
		T d2edy2 = Sqr(this->transfer_function_.dydx(this->out()[no]))*this->d2e_dx2_[no];
		for (UINT ni = 0; ni < this->weights().h(); ++ni) {
			//Accumulate here. Should be called average before use
			this->d2e_dw2_(no,ni) += d2edy2*Sqr(flat_input[ni]);
			flat_d2edx2_prev[ni] += Sqr(this->weights_(no,ni))*d2edy2;
		}
		this->d2e_db2_[no] += d2edy2;//[no];
	}
	this->num_hessian_accums_++;
}

template<class T, class TF>
template<bool hessian>
void FLayer<Tensor,T, TF>::ComputeDerrivativeTemplate(const Tensor<T>& input)
{
    //Flatten the output of previous layer
    Tensor<T> flat_input(input, true);
    flat_input.Flatten();

	assert(this->de_dx_.num_elements() == this->out().num_elements());
	assert(this->de_dw_.HaveSameSize(this->weights()));
	assert(this->de_db_.HaveSameSize(this->biases()));

	const Tensor<T>& de_dx_t = hessian ? this->d2e_dx2() : this->de_dx();
	Tensor<T>& de_dw_t = hessian ? this->d2e_dw2_ : this->de_dw_;
	Tensor<T>& de_db_t = hessian ? this->d2e_db2_ : this->de_db_;

	for (UINT no = 0; no < this->out().num_elements(); ++no) {
		T dedy = this->transfer_function_.dydx(this->out()[no])*de_dx_t[no];
		for (UINT ni = 0; ni < this->weights().h(); ++ni) {
			de_dw_t(no,ni) = dedy*flat_input[ni];
		}
		de_db_t[no] = dedy;
	}
}


/* Compute gradient without backpropagating errors */
template<class T, class TF>
void FLayer<Tensor,T, TF>::ComputeGradient(const Tensor<T>& input)
{
    //Flatten the output of previous layer
    Tensor<T> flat_input(input, true);
    flat_input.Flatten();
	ComputeDerrivativeTemplate<false>(flat_input);
}
/* Compute Hessian without backpropagating errors */
template<class T, class TF>
void FLayer<Tensor,T, TF>::ComputeHessian(const Tensor<T>& input)
{
    //Flatten the output of previous layer
    Tensor<T> flat_input(input, true);
    flat_input.Flatten();

	ComputeDerrivativeTemplate<true>(flat_input);
	this->num_hessian_accums_++;
}

template<class T, class TF>
void FLayer<Tensor,T,TF>::AverageHessian()
{
	if(this->num_hessian_accums_){
		for(UINT i = 0; i < this->d2e_dw2_.num_elements(); ++i){
			this->d2e_dw2_[i] /= this->num_hessian_accums_;
		}
		for(UINT i = 0; i < this->d2e_db2_.num_elements(); ++i){
			this->d2e_db2_[i] /= this->num_hessian_accums_;
		}
	}
	this->num_hessian_accums_ = 0;
}
template<class T, class TF>
void FLayer<Tensor,T,TF>::AdaptWeights(T tau, bool use_hessian, T mu)
{
	if(use_hessian){ //Levenberg-Marquardt
		for(UINT i = 0; i < this->weights().num_elements(); ++i) {
			this->weights_[i] -= this->de_dw()[i]*tau/(this->d2e_dw2()[i] + mu);
		}
		for(UINT i = 0; i < this->biases().num_elements(); ++i) {
			this->biases_[i] -= this->de_db()[i]*tau/(this->d2e_dw2()[i] + mu);
		}
	}else{
		for(UINT i = 0; i < this->weights().num_elements(); ++i) {
			this->weights_[i] -= this->de_dw()[i]*tau;
		}
		for(UINT i = 0; i < this->biases().num_elements(); ++i) {
			this->biases_[i] -= this->de_db()[i]*tau;
		}
	}
}

}