#ifndef _PERFORMANCE_FUNCTIONS_H
#define _PERFORMANCE_FUNCTIONS_H

namespace cudacnn
{

enum ePerfType
{
    eMSE,
    eSSE
};

template <class T> class PerformanceFunctionBase;
template <class T> class MSEFunction;
template <class T> class SSEFunction;


//Factory class
template <class T> 
class PerformanceFunction
{
public:
	PerformanceFunction(ePerfType pf) : performance_function_(NULL) 
	{
		Create(pf);
	}
	void Create(ePerfType pf)
	{
		perf_type_ = pf;
		if(performance_function_) delete performance_function_;
		switch(pf)
		{
		case eSSE:
			performance_function_ = new SSEFunction<T>();
			break;
		case eMSE:
		default:
			performance_function_ = new MSEFunction<T>();
			break;
		}
	}
	~PerformanceFunction()
	{
		if(performance_function_) delete performance_function_;
	}
	PerformanceFunction<T>& operator =(const PerformanceFunction<T>& rhs )
	{
		if (this == &rhs)      // Same object?
			return *this;
		Create(rhs.type());
		return *this;
	}
	T operator() (const Tensor<T>& x)
	{
		return (*performance_function_)(x);
	}
	Tensor<T> dydx(const Tensor<T>& x)
	{
		return performance_function_->dydx(x);
	}
	Tensor<T> d2ydx2(const Tensor<T>& x)
	{
		return performance_function_->d2ydx2(x);
	}
	ePerfType type() const { return perf_type_; }

protected:
	PerformanceFunctionBase<T>* performance_function_;
	ePerfType perf_type_;
};

//Base class
template <class T>
class PerformanceFunctionBase
{
public:
	virtual T operator() (const Tensor<T>& x) = 0;
	virtual Tensor<T> dydx(const Tensor<T>& x) = 0;
	virtual Tensor<T> d2ydx2(const Tensor<T>& x) = 0;
};

template <class T>
class MSEFunction : public PerformanceFunctionBase<T>
{
public:
	inline T operator() (const Tensor<T>& e)
	{
		T out = 0;
		for(UINT i = 0; i < e.num_elements();  ++i)
			out += e[i]*e[i];
		out /= e.num_elements();
		return out;
	}
	inline Tensor<T> dydx(const Tensor<T>& y)
	{
		Tensor<T> out = y;
		for(UINT i = 0; i < y.num_elements(); ++i)
			out[i] = T(2.)*y[i]/y.num_elements();
		return out;
	}
	inline Tensor<T> d2ydx2(const Tensor<T>& y)
	{
		//return Tensor<T>::Ones(y.num_elements()); 
		Tensor<T> out = y;
		for(UINT i = 0; i < y.num_elements(); ++i)
			out[i] = T(2.)/y.num_elements();
            //out[i] = T(2.);
		return out;
	}
};

template <class T>
class SSEFunction 	: public PerformanceFunctionBase<T>
{
public:
	inline T operator() (const Tensor<T>& x)
	{
		std::runtime_error("Not implemented");
		return T(0);
	}
	inline Tensor<T> dydx(const Tensor<T>& y)
	{
		std::runtime_error("Not implemented");
		return Tensor<T>();
	}
	inline Tensor<T> d2ydx2(const Tensor<T>& y)
	{
		std::runtime_error("Not implemented");
		return Tensor<T>();
	}
};

}

#endif