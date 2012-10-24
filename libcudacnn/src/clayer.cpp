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

template class CLayer<Tensor, float, TansigMod<float> >;
template class CLayer<Tensor, float, Tansig<float> >;
template class CLayer<Tensor, float, Purelin<float> >;
template class CLayer<Tensor, double, TansigMod<double> >;
template class CLayer<Tensor, double, Tansig<double> >;
template class CLayer<Tensor, double, Purelin<double> >;

//Naive implementation. Not optimized
template<class T, class TF>
void CLayer<Tensor,T, TF>::Conv2Valid(const Tensor<T>& input, const Tensor<T>& kernel, Tensor<T>& output)
{
	//For 2d data only
	assert(input.num_dims() <=2 );
	assert(kernel.num_dims() <=2 );
	assert(output.num_dims() <=2 );
	assert(output.w() < input.w());
	assert(output.h() < input.h());
	for (unsigned y = 0; y < output.h(); ++y) {
		for (unsigned x = 0; x < output.w(); ++x) {
			//Kernel
			for (unsigned kx = 0; kx < kernel.w(); ++kx) {
				for (unsigned ky = 0; ky < kernel.h(); ++ky) {
					output(x,y) = input(x+kx, y+ky) * kernel(kx,ky);
				}
			}					
		}
	}

}


template<class T, class TF>
void CLayer<Tensor,T, TF>::Propagate(const Tensor<T>& input)
{
	//throw std::runtime_error("Not implemented");
	this->out_.ZeroMemory();
	//Shortcut
	const Tensor<T>& wm = this->weights(); //Weights is 4-dim tensor
	for (unsigned no = 0; no < this->weights().d2(); ++no) {
		for (unsigned ni = 0; ni < input.d(); ++ni) {
			//Output
			if(this->con_map_[ni + no*input.d()]== 0) continue;  //Skip
			for (unsigned y = 0; y < this->out_.h(); ++y) {
				for (unsigned x = 0; x < this->out_.w(); ++x) {
					//Kernel
					for (unsigned kx = 0; kx < wm.w(); ++kx) {
						for (unsigned ky = 0; ky < wm.h(); ++ky) {
							this->out_(x,y,no) += input(x+kx, y+ky, ni) * wm(kx,ky,ni,no);
						}
					}					
				}
			}
		}
		for (unsigned y = 0; y < this->out_.h(); ++y) {
			for (unsigned x = 0; x < this->out_.w(); ++x) {
				this->out_(x,y,no) += this->biases_[no]; 
				this->out_(x,y,no) = this->transfer_function_(this->out_(x,y,no));
			}
		}

	}
};


template<class T, class TF>
void CLayer<Tensor,T, TF>::BackPropagate(const Tensor<T>& input, Tensor<T>& dedx_prev)
{
	BackPropagateTemplate<false>(input, dedx_prev);
}

template<class T, class TF>
template<bool hessian>
void CLayer<Tensor,T, TF>::BackPropagateTemplate(const Tensor<T>& input, Tensor<T>& de_dx_prev)
{
	Tensor<T>& de_dw_t = hessian ? this->d2e_dw2_ : this->de_dw_;
	Tensor<T>& de_db_t = hessian ? this->d2e_db2_ : this->de_db_;
	Tensor<T>& de_dx_t = hessian ? this->d2e_dx2_ : this->de_dx_;
	assert(de_dw_t.HaveSameSize(this->weights()));
	assert(de_db_t.HaveSameSize(this->biases()));
	assert(de_dx_t.HaveSameSize(this->out()));

	if(!hessian){
		de_dw_t.ZeroMemory();
		de_db_t.ZeroMemory();
	}
	de_dx_prev.ZeroMemory();

	for (UINT no = 0; no < this->weights().d2(); ++no) {
		for (UINT ni = 0; ni < input.d(); ++ni) {
			//dEdB independent from connection matrix

			if(this->con_map_[ni + no*input.d()]== 0) continue;  //Skip

			for (unsigned y = 0; y < de_dx_t.h(); ++y) {
				for (unsigned x = 0; x < de_dx_t.w(); ++x) {
					T dedy = hessian ? Sqr(this->transfer_function_.dydx(this->out_(x,y,no)))*de_dx_t(x,y,no) : 
						this->transfer_function_.dydx(this->out_(x,y,no))*de_dx_t(x,y,no);
					for (int ky = this->weights_.h() - 1; ky >= 0; --ky) {
						for (int kx = this->weights_.w() - 1; kx >= 0; --kx) {
							//Calc gradients
							T inp = hessian ? Sqr(input(x+kx, y+ky, ni)) : input(x+kx, y+ky, ni);

							de_dw_t(kx, ky, ni, no) += inp * dedy;							
							de_dx_prev(x + kx, y + ky, ni) += hessian ?  dedy*Sqr(this->weights_(kx,ky,ni, no)) : 
								dedy*this->weights_(kx,ky,ni,no);							
						} //kx
					} //ky
				} //x
			} //y
		}
		for (unsigned y = 0; y < de_dx_t.h(); ++y) {
			for (unsigned x = 0; x < de_dx_t.w(); ++x) {
				T dedy = hessian ? Sqr(this->transfer_function_.dydx(this->out_(x,y,no)))*de_dx_t(x,y,no) : 
					this->transfer_function_.dydx(this->out_(x,y,no))*de_dx_t(x,y,no);
				de_db_t[no] += dedy;
			}
		}

	}
}

template<class T, class TF>
template<bool hessian>
void CLayer<Tensor,T, TF>::ComputeDerrivativeTemplate(const Tensor<T>& input)
{
	Tensor<T>& de_dw_t = hessian ? this->d2e_dw2_ : this->de_dw_;
	Tensor<T>& de_db_t = hessian ? this->d2e_db2_ : this->de_db_;
	Tensor<T>& de_dx_t = hessian ? this->d2e_dx2_ : this->de_dx_;
	assert(de_dw_t.HaveSameSize(this->weights()));
	assert(de_db_t.HaveSameSize(this->biases()));
	assert(de_dx_t.HaveSameSize(this->out()));

	if(!hessian){
		de_dw_t.ZeroMemory();
		de_db_t.ZeroMemory();
	}

	for (UINT no = 0; no < this->weights().d2(); ++no) {
		for (UINT ni = 0; ni < input.d(); ++ni) {
			//dEdB independent from connection matrix

			if(this->con_map_[ni + no*input.d()]== 0) continue;  //Skip

			for (unsigned y = 0; y < de_dx_t.h(); ++y) {
				for (unsigned x = 0; x < de_dx_t.w(); ++x) {
					T dedy = hessian ? Sqr(this->transfer_function_.dydx(this->out_(x,y,no)))*de_dx_t(x,y,no) : 
						this->transfer_function_.dydx(this->out_(x,y,no))*de_dx_t(x,y,no);
				for (int ky = this->weights_.h() - 1; ky >= 0; --ky) {
					for (int kx = this->weights_.w() - 1; kx >= 0; --kx) {
						//Calc gradients
						T inp = hessian ? Sqr(input(x+kx, y+ky, ni)) : input(x+kx, y+ky, ni);
						de_dw_t(kx, ky, ni, no) += inp * dedy;							
					} //kx
				} //ky
				} //x
			} //y
		}
		for (unsigned y = 0; y < de_dx_t.h(); ++y) {
			for (unsigned x = 0; x < de_dx_t.w(); ++x) {
				T dedy = hessian ? Sqr(this->transfer_function_.dydx(this->out_(x,y,no)))*de_dx_t(x,y,no) : 
					this->transfer_function_.dydx(this->out_(x,y,no))*de_dx_t(x,y,no);
			de_db_t[no] += dedy;
			}
		}

	}
}

template<class T, class TF>
void CLayer<Tensor,T, TF>::BackPropagateHessian(const Tensor<T>& input, Tensor<T>& d2edx2_prev) 
{
	BackPropagateTemplate<true>(input, d2edx2_prev);
	this->num_hessian_accums_++;
}

/* Compute gradient without backpropagating errors */
template<class T, class TF>
void CLayer<Tensor,T, TF>::ComputeGradient(const Tensor<T>& input)
{
	ComputeDerrivativeTemplate<false>(input);
}
/* Compute Hessian without backpropagating errors */
template<class T, class TF>
void CLayer<Tensor,T, TF>::ComputeHessian(const Tensor<T>& input)
{
	ComputeDerrivativeTemplate<true>(input);
	this->num_hessian_accums_++;
}

template<class T, class TF>
void CLayer<Tensor,T, TF>::AverageHessian()
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
void CLayer<Tensor,T, TF>::AdaptWeights(T tau, bool use_hessian, T mu)
{
	if(use_hessian){
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