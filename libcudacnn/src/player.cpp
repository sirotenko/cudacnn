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

//Instantiate
template class PoolingLayer<Tensor,float>;
template class PoolingLayer<Tensor,double>;

//Simple implementation. Mainly for testing
template<class T>
void PoolingLayer<Tensor,T>::PropagateAverage(const Tensor<T>& input)
{
        for(unsigned n = 0; n < out_.d(); n++){
            for(unsigned y = 0; y < out_.h(); y++){
                for(unsigned x = 0; x < out_.w(); x++) {
                    T sum = 0;
                    for(UINT syi = 0; syi < sy_; ++syi){
                        for(UINT sxi = 0; sxi < sx_; ++sxi){
                            sum += input(x*sx_+sxi, y*sy_+syi, n);
                        }
                    }
                    out_(x,y,n) = sum/(sy_*sx_);
                }
            }
        }
}
template<class T>
void PoolingLayer<Tensor,T>::PropagateMax(const Tensor<T>& input)
{
    for(unsigned n = 0; n < out_.d(); n++){
        for(unsigned y = 0; y < out_.h(); y++){
            for(unsigned x = 0; x < out_.w(); x++) {
                T max_val = input(x*sx_, y*sy_, n); // first elem
                for(UINT syi = 0; syi < sy_; ++syi){
                    for(UINT sxi = 0; sxi < sx_; ++sxi){
                        max_val = std::max(max_val, input(x*sx_+sxi, y*sy_+syi, n));
                    }
                }
                out_(x,y,n) = max_val;
            }
        }
    }
}


template<class T>
void PoolingLayer<Tensor,T>::Propagate(const Tensor<T>& input)
{
	assert(out_.w() == input.w()/sx_ && out_.h() == input.h()/sy_);
    switch(pooling_type_){
        case eAverage:
            PropagateAverage(input);
            break;
        case eMax:
            PropagateMax(input);
            break;
        default:
            throw std::runtime_error("Unknown pooling type");
    }    
}



template<class T>
template<bool hessian>
void PoolingLayer<Tensor,T>::BackPropagateTemplateAverage(const Tensor<T>& input, Tensor<T>& dedx_prev)
{
	Tensor<T>& de_dx_t = hessian ? d2e_dx2_ : de_dx_;

	assert(dedx_prev.HaveSameSize(input));

	for(unsigned n = 0; n < out_.d(); n++){
		for(unsigned y = 0; y < out_.h(); y++){
			for(unsigned x = 0; x < out_.w(); x++) {
				for(UINT syi = 0; syi < sy_; ++syi){
					for(UINT sxi = 0; sxi < sx_; ++sxi){
						dedx_prev(x*sx_+sxi, y*sy_+syi, n) = de_dx_t(x,y,n)/(sy_*sx_);
					}
				}
			}
		}
	}
}
template<class T>
template<bool hessian>
void PoolingLayer<Tensor,T>::BackPropagateTemplateMax(const Tensor<T>& input, Tensor<T>& dedx_prev)
{
    Tensor<T>& de_dx_t = hessian ? d2e_dx2_ : de_dx_;

    assert(dedx_prev.HaveSameSize(input));

    for(unsigned n = 0; n < out_.d(); n++){
        for(unsigned y = 0; y < out_.h(); y++){
            for(unsigned x = 0; x < out_.w(); x++) {
                for(UINT syi = 0; syi < sy_; ++syi){
                    for(UINT sxi = 0; sxi < sx_; ++sxi){
                        dedx_prev(x*sx_+sxi, y*sy_+syi, n) =  
                            //Check if this is the input corresponding to max out
                            input(x*sx_+sxi, y*sy_+syi, n) == out_(x,y,n) ? de_dx_t(x,y,n) : 0;
                        //TODO: some issues with floats equality test?
                    }
                }
            }
        }
    }
}



template <class T>
void PoolingLayer<Tensor,T>::BackPropagateHessian(const Tensor<T>& input, Tensor<T>& d2edx2_prev)
{
    switch(pooling_type_){
        case eAverage:
            BackPropagateTemplateAverage<true>(input, d2edx2_prev);
            break;
        case eMax:
            BackPropagateTemplateMax<true>(input, d2edx2_prev);
            break;
        default:
            throw std::runtime_error("Unknown pooling type");
    }  
	
	//Don't increment since we are not using weignts in this implementation of S-Layer
	//num_hessian_accums_++;
}

template <class T>
void PoolingLayer<Tensor,T>::BackPropagate(const Tensor<T>& input, Tensor<T>& dedx_prev)
{
    switch(pooling_type_){
        case eAverage:
            BackPropagateTemplateAverage<false>(input, dedx_prev);
            break;
        case eMax:
            BackPropagateTemplateMax<false>(input, dedx_prev);
            break;
        default:
            throw std::runtime_error("Unknown pooling type");
    }  
	
}

}


