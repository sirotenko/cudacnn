#include "precomp.hpp"


#ifdef __CUDACNNMEX_FLOAT
typedef float BaseType;
#endif

#ifdef __CUDACNNMEX_DOUBLE
typedef double BaseType;
#endif

#ifdef __CUDACNNMEX_CUDA 
#define BaseTensor TensorGPU
#endif

#ifdef __CUDACNNMEX_CPU
#define BaseTensor Tensor
#endif


static CNNet<BaseTensor, BaseType>* cnn = NULL;
static Trainer<BaseTensor, BaseType>* cnn_trainer = NULL;

template< template<class> class MAT, class T>
Trainer<MAT, T>* InitTrainer(CNNet<MAT, T>* cnn, const mxArray * trainer_arr ) 
{
	if (trainer_arr == NULL) {
		throw std::runtime_error("Trainer structure array is null");
	}
	if (mxIsEmpty(trainer_arr)) {
		std::stringstream ss;
		ss<<"InitTrainer failed"<<std::endl;
		ss<<"Trainer structure should be non empty";
		throw std::runtime_error(ss.str());
	}
	if (!mxIsStruct(trainer_arr)){
		std::stringstream ss;
		ss<<"InitTrainer failed"<<std::endl;
		ss<<"Trainer structure should be struct";
		throw std::runtime_error(ss.str());
	}
	
	const char* train_method = GetSVal(trainer_arr, "TrainMethod");
	if(strcmp(train_method,"StochasticLM")==0) { //Stochastic Levenberg Marquardt
		StochasticLMTrainer<MAT, T>* trainer = new StochasticLMTrainer<MAT,T>(cnn);
		trainer->settings.epochs = GetScalar<UINT>(trainer_arr,"epochs");
		trainer->settings.mu = GetScalar<T>(trainer_arr,"mu");
		Tensor<T> theta_tmp = GetArray<T>(trainer_arr,"theta");
		if(theta_tmp.num_elements() != trainer->settings.epochs){
			throw std::runtime_error("InitTrainer failed. Number of elements in theta should be equal to number of epochs.");
		}
		//trainer->settings.theta = std::vector(trainer->settings.epochs,0);
		trainer->settings.theta = std::vector<T>(theta_tmp.data(),theta_tmp.data()+theta_tmp.num_elements()*sizeof(T));
		return trainer;
	}
	else if(strcmp(train_method,"StochasticGD")==0){ //Stochastic Gradient descent
		StochasticGDTrainer<MAT, T>* trainer = new StochasticGDTrainer<MAT,T>(cnn);
		trainer->settings.epochs = GetScalar<UINT>(trainer_arr,"epochs");
		Tensor<T> theta_tmp = GetArray<T>(trainer_arr,"theta");
		if(theta_tmp.num_elements() != trainer->settings.epochs){
			throw std::runtime_error("InitTrainer failed. Number of elements in theta should be equal to number of epochs.");
		}
		//trainer->settings.theta = std::vector(trainer->settings.epochs,0);
		trainer->settings.theta = std::vector<T>(theta_tmp.data(),theta_tmp.data()+theta_tmp.num_elements()*sizeof(T));
		return trainer;
	}
	else throw std::runtime_error("InitTrainer failed. Unknown train method.");

}


//Display CUDA device information
void CudaInfoMatlab()
{
	cudaError_t err = cudaSuccess;
	int            deviceCount;
	cudaDeviceProp devProp;

	err = cudaGetDeviceCount ( &deviceCount );
	if(!deviceCount) mexErrMsgTxt("Error: No CUDA devices found!");

	mexPrintf ( "Found %d devices\n", deviceCount );

	for ( int device = 0; device < deviceCount; device++ )
	{
		cudaGetDeviceProperties ( &devProp, device );

		mexPrintf ( "Device %d\n", device );
		mexPrintf ( "Compute capability     : %d.%d\n", devProp.major, devProp.minor );
		mexPrintf ( "Name                   : %s\n", devProp.name );
		mexPrintf ( "Total Global Memory    : %d\n", devProp.totalGlobalMem );
		mexPrintf ( "Shared memory per block: %d\n", devProp.sharedMemPerBlock );
		mexPrintf ( "Registers per block    : %d\n", devProp.regsPerBlock );
		mexPrintf ( "Warp size              : %d\n", devProp.warpSize );
		mexPrintf ( "Max threads per block  : %d\n", devProp.maxThreadsPerBlock );
		mexPrintf ( "Total constant memory  : %d\n", devProp.totalConstMem );
	}

}


void cleanup(void)
{
	if(cnn)
	{
		delete cnn;
		delete cnn_trainer;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{

	// Check inputs
	if (nrhs <1) mexErrMsgTxt("Must have at least two arguments: cnnet object and operation mode");
	if ( !mxIsClass(prhs[0],"cnn")) mexErrMsgTxt("First input must be a convolutional neural network object");
	/* input must be a string */
	if ( mxIsChar(prhs[1]) != 1)
		mexErrMsgTxt("Second argument must be a string.");
	/* input must be a row vector */
	if (mxGetM(prhs[1])!=1)
		mexErrMsgTxt("Input must be a row vector.");
	/* get the length of the input string */
	mwSize buflen = (mxGetM(prhs[1]) * mxGetN(prhs[1])) + 1;
	/* copy the string data from prhs[0] into a C string input_ buf.    */
	char* input_buf = mxArrayToString(prhs[1]);
	if(strcmp(input_buf,"init")==0)	{
		if (nrhs != 2) {
			std::stringstream ss;
			ss<<"Not enough input arguments. Usage: \n";
			ss<<"cudacnnMex(cnn,'init'), where \n";
			ss<<"cnn - convolutional neural network class object \n";
			mexErrMsgTxt(ss.str().c_str());
		}
		delete cnn;
		cnn = new CNNet<BaseTensor, BaseType>();
#ifdef __CUDACNNMEX_CUDA
		CudaInfoMatlab();
#endif
		try		{
			CNNetLoadFromMatlab(prhs[0],*cnn);
		}
		catch(CudaException& e)		{
			mexErrMsgTxt(e.what());
		}
		//TODO: Reconsider this to be called only when training is needed.
		cnn->PrepareForTraining();
		mexAtExit(cleanup);
		mexPrintf("CNN successfully initialized. \n");
	} 
	else if(strcmp(input_buf,"save")==0)
	{
		if (nlhs !=1) mexErrMsgTxt("Must have one output: structure of convolutional neural network");
		if (cnn == NULL) mexErrMsgTxt("Nothing to save. No network initialized");

		plhs[0] = CNNetSaveToMatlab(*cnn);
	}
	else if(strcmp(input_buf,"debug_save")==0)
	{
		if (nlhs !=1) mexErrMsgTxt("Must have one output: structure of convolutional neural network");
		if (cnn == NULL) mexErrMsgTxt("Nothing to save. No network initialized");

		plhs[0] = CNNetSaveToMatlab(*cnn, true);
	}
	else if(strcmp(input_buf,"sim")==0)	{
        
		if(!cnn) mexErrMsgTxt("CNN must be initialized first");
		if (nrhs != 3) {
			std::stringstream ss;
			ss<<"Not enough input arguments. Usage: \n";
			ss<<"cudacnnMex(cnn, 'sim',input), where \n";
			ss<<"cnn - convolutional neural network class object \n";
			ss<<"input - matrix or 3D array of inputs \n";
			mexErrMsgTxt(ss.str().c_str());
		}
		if (nlhs !=1) mexErrMsgTxt("Must have one output argument");
		if ( mxIsComplex(prhs[2]) || !mxIsClass(prhs[2],"single"))	mexErrMsgTxt("Second Input must be real, single type");
		try		{
			TensorFloat inp;
			MatlabTools::GetArray<BaseType>(prhs[2], inp);
			cnn->Sim(inp);
		}
		catch(CudaException& e)	{
			mexErrMsgTxt(e.what());
		}
		catch(std::runtime_error& e) {
			mexErrMsgTxt(e.what());
		}
		
		TensorFloat out = cnn->out();
        plhs[0] = mxCreateNumericMatrix(out.w(),out.h(),mxSINGLE_CLASS,mxREAL);
        float *outputX = (float*)mxGetData(plhs[0]);
        memcpy(outputX,out.data(),out.num_elements()*sizeof(float)); 	

	}
	else if(strcmp(input_buf,"accum_hessian")==0) {
		if(!cnn) mexErrMsgTxt("CNN must be initialized first");
		if (nrhs != 4) {
			std::stringstream ss;
			ss<<"Not enough input arguments. Usage: \n";
			ss<<"cudacnnMex(cnn, 'accum_hessian', err, inp), where \n";
			ss<<"cnn - convolutional neural network class object \n";
			ss<<"err - errors vector \n";
			ss<<"inp - network input \n";
			mexErrMsgTxt(ss.str().c_str());
		}
		BaseTensor<BaseType> err;
		GetArray<BaseType>(prhs[2], err);
		if(err.num_elements() != cnn->out().num_elements()){
			mexErrMsgTxt("Number of error vector elements should be equal to the number of network outputs");
		}
		BaseTensor<BaseType> inp;
		GetArray<BaseType>(prhs[3], inp);
		cnn->AccumulateHessian(err, inp);
	}
	else if(strcmp(input_buf,"average_hessian")==0) {
		if(!cnn) mexErrMsgTxt("CNN must be initialized first");
		if (nrhs != 2) {
			std::stringstream ss;
			ss<<"Not enough input arguments. Usage: \n";
			ss<<"cudacnnMex(cnn, 'average_hessian'), where \n";
			ss<<"cnn - convolutional neural network class object \n";
			mexErrMsgTxt(ss.str().c_str());
		}
		cnn->AverageHessian();
	}
	else if(strcmp(input_buf,"compute_gradient")==0) {
		if(!cnn) mexErrMsgTxt("CNN must be initialized first");
		if (nrhs != 4) {
			std::stringstream ss;
			ss<<"Not enough input arguments. Usage: \n";
			ss<<"cudacnnMex(cnn, 'compute_gradient',err,inp), where \n";
			ss<<"cnn - convolutional neural network class object \n";
			ss<<"err - errors vector \n";
			ss<<"inp - network input \n";
			mexErrMsgTxt(ss.str().c_str());
		}
		BaseTensor<BaseType> err;
		GetArray<BaseType>(prhs[2], err);
		if(err.num_elements() != cnn->out().num_elements()){
			mexErrMsgTxt("Number of error vector elements should be equal to the number of network outputs");
		}
		BaseTensor<BaseType> inp;
		GetArray<BaseType>(prhs[3], inp);
		cnn->BackpropGradients(err, inp);
	}
	else if(strcmp(input_buf,"adapt")==0) {
		if(!cnn) mexErrMsgTxt("CNN must be initialized first");
		if (nrhs != 5) {
			std::stringstream ss;
			ss<<"Not enough input arguments. Usage: \n";
			ss<<"cudacnnMex(cnn, 'compute_gradient', theta, use_hessian, mu), where \n";
			ss<<"cnn - convolutional neural network class object \n";
			ss<<"theta - train coefficient \n";
			ss<<"use_hessian - if 1, use hessian for weights change, otherwise use only gradients \n";
			ss<<"mu - if use hessian, small constant preventing blow up of train coefficient in case of small hessian \n";
			mexErrMsgTxt(ss.str().c_str());
		}
		BaseType tau = GetScalar<BaseType>(prhs[2]);
		bool use_hessian = GetScalar<UINT>(prhs[3]) == 1 ? true : false;
		BaseType mu = GetScalar<BaseType>(prhs[4]);
		cnn->AdaptWeights(tau,use_hessian, mu);
	}
	else if(strcmp(input_buf,"destroy")==0) {
		if (cnn == NULL)
			mexWarnMsgTxt("Nothing to Destroy. Network not initialized. \n");
		else
		{
			delete cnn;
			cnn = NULL;
		}
		//mexPrintf("CNN successfully destroyed. \n");
	}
	else {
		mexErrMsgTxt("Unknown command.");
	}
    mxFree(input_buf);

}




