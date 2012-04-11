#ifndef _TRANSFER_FUNCTIONS_H
#define _TRANSFER_FUNCTIONS_H

//template <class T> class TransferFunctionBase;
//template <class T> class Purelin;
//template <class T> class Tansig;
//template <class T> class TansigMod;


//Factory class
//template <class T> 
//class TransferFunction
//{
//public:
//	TransferFunction(eTransfFunc tf) : transfer_function_(NULL) 
//	{
//		Create(tf);
//	}
//	void Create(eTransfFunc tf)
//	{
//		transfer_type_ = tf;
//		if(transfer_function_) delete transfer_function_;
//		switch(tf)
//		{
//		case ePurelin:
//			transfer_function_ = new Purelin<T>();
//			break;
//		case eTansig:
//			transfer_function_ = new Tansig<T>();
//			break;
//		case eTansig_mod:
//			transfer_function_ = new TansigMod<T>();
//			break;
//		default:
//			transfer_function_ = new Purelin<T>();
//		}
//	}
//	~TransferFunction()
//	{
//		if(transfer_function_) delete transfer_function_;
//	}
//	TransferFunction<T>& operator =(const TransferFunction<T>& rhs )
//	{
//		if (this == &rhs)      // Same object?
//			return *this;
//		Create(rhs.type());
//		return *this;
//	}
//	T operator() (T x)
//	{
//		return (*transfer_function_)(x);
//	}
//	T dydx(T fn_out)
//	{
//		return transfer_function_->dydx(fn_out);
//	}
//	T d2ydx2(T fn_out)
//	{
//		return transfer_function_->d2ydx2(fn_out);
//	}
//	eTransfFunc type() const { return transfer_type_; }
//	std::string name() const { return transfer_function_->name(); }
//protected:
//	TransferFunctionBase<T>* transfer_function_;
//	eTransfFunc transfer_type_;
//};
//
//Base class
//template <class T>
//class TransferFunctionBase
//{
//public:
//	virtual T operator() (T x) {return 0;};
//	virtual T dydx(T fn_out) {return 0;};
//	virtual T d2ydx2(T fn_out) {return 0;};
//	virtual std::string name() {return 0;};
//};

template <class T>
class TansigMod //: public TransferFunctionBase<T>
{
public:
	__device__ T operator() (T x)
	{
		return T(1.7159*tanh(0.66666667*x));
	}
	__device__ T dydx(T fn_out)
	{
		T x = T(0.66666667/1.7159*(1.7159+fn_out)*(1.7159-fn_out));
		return x; 

	}
	__device__ T d2fn_outdx2(T fn_out)
	{
		T x = T(0.66666667/1.7159*(1.7159+fn_out)*(1.7159-fn_out));
		return x*x; 
	}
	std::string name() { return "tansig_mod";}
};

template <class T>
class Purelin //: public TransferFunctionBase<T>
{
public:
	__device__ T operator() (T x)
	{
		return x;
	}
	__device__ T dydx(T fn_out)
	{
		return 1; 
	}
	__device__ T d2ydx2(T fn_out)
	{
		return 0; 
	}
	std::string name() { return "purelin";}
};

template <class T>
class Tansig //: public TransferFunctionBase<T>
{
public:
	__device__ T operator() (T x)
	{
		return tanh(x);
	}
	__device__ T dydx(T fn_out)
	{
		return (1-fn_out*fn_out); 
	}
	__device__ T d2ydx2(T fn_out)
	{
		return (1-fn_out*fn_out)*(1-fn_out*fn_out); 
	}
	std::string name() { return "tansig";}
};


#endif
