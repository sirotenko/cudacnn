#ifndef _TRAINER_H
#define _TRAINER_H
#include "common.h"
#include "tensor.h"
#include "conv_net.h"

//Trainer base class
template< template<class> class MAT, class T>
class Trainer
{
public:
	Trainer(CNNet<MAT, T>* cnnet):cnnet_(cnnet) {};
	//Train over all epochs, all iterations, all data
	//void Train() = 0;
	//Only apply 1 step of weights change
	virtual void Adapt(T tau) = 0;
protected:
	CNNet<MAT, T>* cnnet_;
};

//Stochastic Levenberg-Marquardt trainer
template< template<class> class MAT, class T>
class StochasticLMTrainer : public Trainer<MAT, T>
{
public:
	struct Settings
	{
		T mu;
		std::vector<T> theta;
		UINT epochs;
		Settings():mu(T(0.0001f)),epochs(10), theta(std::vector<T>(epochs,0.001f)){};
	};
	Settings settings;
	StochasticLMTrainer(CNNet<MAT, T>* cnnet):Trainer<MAT,T>(cnnet) {};
	//Train over all epochs, all iterations, all data
	void Train();
	//Only apply 1 step of weights change
	void Adapt(T tau);
	void ComputeGradient(const MAT<T>& error);
	void EstimateHessian(const std::vector<MAT<T> >& train_set);
};

template< template<class> class MAT, class T>
void StochasticLMTrainer<MAT,T>::ComputeGradient(const MAT<T>& error)
{
	cnnet_->BackpropGradients(error);
}

template< template<class> class MAT, class T>
void StochasticLMTrainer<MAT,T>::EstimateHessian(const std::vector<MAT<T> >& train_set)
{
	cnnet_->ResetHessian();
	for(train_set::const_iterator it = train_set.begin(); it!= train_set.end(); ++it){
		cnnet_->AccumulateHessian(*it);
	}
	cnnet_->AverageHessian();
}

template< template<class> class MAT, class T>
void StochasticLMTrainer<MAT,T>::Adapt(T tau)
{
	cnnet_->AdaptWeights(tau, true, settings.mu);
}

//Stochastic Gradient descent trainer
template< template<class> class MAT, class T>
class StochasticGDTrainer : public Trainer<MAT, T>
{
public:
	struct Settings
	{
		std::vector<T> theta;
		UINT epochs;
		Settings(): epochs(10), theta(std::vector<T>(epochs,0.001f)){};
	};
	Settings settings;
	StochasticGDTrainer(CNNet<MAT, T>* cnnet):Trainer<MAT,T>(cnnet) {};
	//Train over all epochs, all iterations, all data
	void Train();
	//Only apply 1 step of weights change
	void Adapt(T tau);
	void ComputeGradient(const MAT<T>& error);
	void EstimateHessian(const std::vector<MAT<T> >& train_set);
};

template< template<class> class MAT, class T>
void StochasticGDTrainer<MAT,T>::Adapt(T tau)
{
	cnnet_->AdaptWeights(tau, false, 0);
}

#endif //_TRAINER_H