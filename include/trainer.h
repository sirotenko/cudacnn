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

#ifndef _TRAINER_H
#define _TRAINER_H

namespace cudacnn
{

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
	virtual ~Trainer() {};
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
	void EstimateHessian(const std::vector< MAT<T> >& train_set);
};

template< template<class> class MAT, class T>
void StochasticLMTrainer<MAT,T>::ComputeGradient(const MAT<T>& error)
{
	this->cnnet_->BackpropGradients(error);
}

template< template<class> class MAT, class T>
void StochasticLMTrainer<MAT,T>::EstimateHessian(const std::vector<MAT<T> >& train_set)
{
	this->cnnet_->ResetHessian();
	typename std::vector<MAT<T> >::const_iterator  it;
	for(it = train_set.begin(); it!= train_set.end(); ++it){
		this->cnnet_->AccumulateHessian(*it);
	}
	this->cnnet_->AverageHessian();
}

template< template<class> class MAT, class T>
void StochasticLMTrainer<MAT,T>::Adapt(T tau)
{
	this->cnnet_->AdaptWeights(tau, true, settings.mu);
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
	this->cnnet_->AdaptWeights(tau, false, 0);
}

#endif //_TRAINER_H

}
