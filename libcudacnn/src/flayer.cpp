#include "precomp.hpp"

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
	out_.ZeroMemory();
    //Flatten the output of previous layer
    Tensor<T> flat_input(input, true);
    flat_input.Flatten();

	assert(flat_input.num_elements() == weights_.h());

	for (UINT n = 0; n < weights_.w(); ++n) {	
		for (UINT i = 0; i < flat_input.num_elements(); ++i)	{
			out_[n] += flat_input[i]*weights_(n,i);
		}
		out_[n] = transfer_function_(out_[n] + biases_[n]);
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

	assert(de_dx_.num_elements() == out().num_elements());
	assert(de_dw_.HaveSameSize(weights()));
	assert(de_db_.HaveSameSize(biases()));
	assert(flat_dedx_prev.num_elements() == flat_input.num_elements());
	flat_dedx_prev.ZeroMemory();
	for (UINT no = 0; no < out().num_elements(); ++no) {
		T dedy = transfer_function_.dydx(out()[no])*de_dx_[no];
		for (UINT ni = 0; ni < weights().h(); ++ni) {
			de_dw_(no,ni) = dedy*flat_input[ni];
			flat_dedx_prev[ni] += weights_(no,ni)*dedy;
		}
		de_db_[no] = dedy;//[no];
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

	assert(d2e_dx2_.num_elements() == out().num_elements());
	assert(d2e_dw2_.HaveSameSize(weights()));
	assert(d2e_db2_.HaveSameSize(biases()));
	assert(flat_d2edx2_prev.num_elements() == flat_input.num_elements());
	flat_d2edx2_prev.ZeroMemory();
	for (UINT no = 0; no < out().num_elements(); ++no) {
		T d2edy2 = Sqr(transfer_function_.dydx(out()[no]))*d2e_dx2_[no];
		for (UINT ni = 0; ni < weights().h(); ++ni) {
			//Accumulate here. Should be called average before use
			d2e_dw2_(no,ni) += d2edy2*Sqr(flat_input[ni]);
			flat_d2edx2_prev[ni] += Sqr(weights_(no,ni))*d2edy2;
		}
		d2e_db2_[no] += d2edy2;//[no];
	}
	num_hessian_accums_++;
}

template<class T, class TF>
template<bool hessian>
void FLayer<Tensor,T, TF>::ComputeDerrivativeTemplate(const Tensor<T>& input)
{
    //Flatten the output of previous layer
    Tensor<T> flat_input(input, true);
    flat_input.Flatten();

	assert(de_dx_.num_elements() == out().num_elements());
	assert(de_dw_.HaveSameSize(weights()));
	assert(de_db_.HaveSameSize(biases()));

	const Tensor<T>& de_dx_t = hessian ? d2e_dx2() : de_dx();
	Tensor<T>& de_dw_t = hessian ? d2e_dw2_ : de_dw_;
	Tensor<T>& de_db_t = hessian ? d2e_db2_ : de_db_;

	for (UINT no = 0; no < out().num_elements(); ++no) {
		T dedy = transfer_function_.dydx(out()[no])*de_dx_t[no];
		for (UINT ni = 0; ni < weights().h(); ++ni) {
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
	num_hessian_accums_++;
}

template<class T, class TF>
void FLayer<Tensor,T,TF>::AverageHessian()
{
	if(num_hessian_accums_){
		for(UINT i = 0; i < d2e_dw2_.num_elements(); ++i){
			d2e_dw2_[i] /= num_hessian_accums_;
		}
		for(UINT i = 0; i < d2e_db2_.num_elements(); ++i){
			d2e_db2_[i] /= num_hessian_accums_;
		}
	}
	num_hessian_accums_ = 0;
}
template<class T, class TF>
void FLayer<Tensor,T,TF>::AdaptWeights(T tau, bool use_hessian, T mu)
{
	if(use_hessian){ //Levenberg-Marquardt
		for(UINT i = 0; i < weights().num_elements(); ++i) {
			weights_[i] -= de_dw()[i]*tau/(d2e_dw2()[i] + mu);
		}
		for(UINT i = 0; i < biases().num_elements(); ++i) {
			biases_[i] -= de_db()[i]*tau/(d2e_dw2()[i] + mu);
		}
	}else{
		for(UINT i = 0; i < weights().num_elements(); ++i) {
			weights_[i] -= de_dw()[i]*tau;
		}
		for(UINT i = 0; i < biases().num_elements(); ++i) {
			biases_[i] -= de_db()[i]*tau;
		}
	}
}