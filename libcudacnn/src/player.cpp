#include "precomp.hpp"

//Instantiate
template PoolingLayer<Tensor,float>;
template PoolingLayer<Tensor,double>;

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
                T max_val = 0;
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




