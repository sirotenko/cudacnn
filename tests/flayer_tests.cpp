#include "precomp.h"

class FLayerTest : public CUDATest {
protected:
	TensorGPUFloat weights, biases, inputs, outputs;
	TensorFloat inputsHost, weightsHost, biasesHost, outputsHost;
	virtual void SetUp()
	{
		CUDATest::SetUp();

		inputs = TensorGPUFloat(120, 1, 1);
		outputs = TensorGPUFloat(84, 1, 1);

		std::vector<UINT> wdims;
		wdims.push_back(84);
		wdims.push_back(120);
		weights = TensorGPUFloat(wdims);

		std::vector<UINT> bdims;
		bdims.push_back(1); // tansigmod num = 1 
		bdims.push_back(84);
		biases = TensorGPUFloat(bdims);  // tansigmod num = 1

		inputsHost = TensorFloat(inputs.dims());
		weightsHost = TensorFloat(weights.dims());
		biasesHost = TensorFloat(biases.dims());
		outputsHost = TensorFloat(outputs.dims());
	}
	virtual void TearDown()
	{
	}
};

TEST_F(FLayerTest, PropagateZeros)
{
	inputsHost = inputs;
	for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
		EXPECT_EQ(inputsHost[i], 0);
	}

	FLayerCudaFTS flayer(weights, biases, true);

	flayer.Propagate(inputs);
	TensorFloat outHost;
	outHost = flayer.out();

	for (unsigned i = 0; i < outHost.num_elements(); i++) {
		ASSERT_EQ(outHost[i], 0);
	}
}

TEST_F(FLayerTest, BackPropZeros)
{
	inputsHost = inputs;
	outputsHost = outputs;

	for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
		EXPECT_EQ(inputsHost[i], 0);
	}
	for (unsigned i = 0; i < outputsHost.num_elements(); i++) {
		EXPECT_EQ(outputsHost[i], 0);
	}
	
	TensorGPUFloat de_dx_prev(inputs);

	FLayerCudaFTS flayer(weights, biases, true);    
	flayer.PrepareForTraining();
	flayer.set_de_dx(outputs);
	flayer.BackPropagate(inputs, de_dx_prev);

    TensorFloat de_dx_prev_host = de_dx_prev;
	for(unsigned i = 0; i < de_dx_prev_host.num_elements(); ++i){
		ASSERT_NEAR(de_dx_prev_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
	TensorFloat de_dw_host = flayer.de_dw();
	for(unsigned i = 0; i < de_dw_host.num_elements(); ++i){
		ASSERT_NEAR(de_dw_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
	TensorFloat de_dp_host = flayer.de_db();
	for(unsigned i = 0; i < de_dp_host.num_elements(); ++i){
		ASSERT_NEAR(de_dp_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
}

TEST_F(FLayerTest, ComputeDerivZeros)
{
	inputsHost = inputs;
	outputsHost = outputs;

	for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
		EXPECT_EQ(inputsHost[i], 0);
	}	
	for (unsigned i = 0; i < outputsHost.num_elements(); i++) {
		EXPECT_EQ(outputsHost[i], 0);
	}
	
	FLayerCudaFTS flayer(weights, biases, true);    
	flayer.PrepareForTraining();
	flayer.set_de_dx(outputs);
	flayer.ComputeGradient(inputs);    
	
	TensorFloat de_dw_host = flayer.de_dw();
	for(unsigned i = 0; i < de_dw_host.num_elements(); ++i){
		ASSERT_NEAR(de_dw_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
	TensorFloat de_dp_host = flayer.de_db();
	for(unsigned i = 0; i < de_dp_host.num_elements(); ++i){
		ASSERT_NEAR(de_dp_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
}

TEST_F(FLayerTest, PropagateOnes)
{
	for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
		inputsHost[i] = 1.f;
	}
	for (unsigned i = 0; i < weightsHost.num_elements(); i++) {
		weightsHost[i] = 1.f;
	}
	for (unsigned i = 0; i < biasesHost.num_elements(); i++) {
		biasesHost[i] = 1.f;
	}

	inputs = inputsHost;
	weights = weightsHost;
	biases = biasesHost;

	FLayerCudaFTS flayer(weights, biases, true);
	flayer.Propagate(inputs);

	TensorFloat out_cuda_host, ws_cuda_host;
	out_cuda_host = flayer.out();

	TansigMod<float> transfer_func;
	
	for (unsigned i = 0; i < out_cuda_host.num_elements(); i++) 
	{
		float true_ws = float(inputs.num_elements());
		float true_out = transfer_func(true_ws);
		ASSERT_EQ(out_cuda_host[i], true_out)<< "outHost is wrong at index " << i;
	}
}

class FLayerRandomTest : public FLayerTest {
protected:
	
	virtual void SetUp()
	{
		FLayerTest::SetUp();	

		for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
			inputsHost[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < weightsHost.num_elements(); i++) {
			weightsHost[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < biasesHost.num_elements(); i++) {
			biasesHost[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < outputsHost.num_elements(); i++) {
			outputsHost[i] = float(std::rand()) / RAND_MAX;
		}

		inputs = inputsHost;
		weights = weightsHost;
		biases = biasesHost;
		outputs = outputsHost;
	}
	virtual void TearDown()
	{
	}
};


TEST_F(FLayerRandomTest, PropagateRandom)
{
	FLayerCudaFTS flayer(weights, biases, true);
	FLayerFTS flayer_host(weights, biases, true);
	
	flayer.Propagate(inputs);
	flayer_host.Propagate(inputs);

	TensorFloat outHost = flayer.out();
	TensorFloat true_out = flayer_host.out();

	for (unsigned i = 0; i < outHost.num_elements(); i++) 
	{
		ASSERT_NEAR(outHost[i], true_out[i], 100.*FLT_EPSILON)<< "outHost is wrong at index " << i;
	}
}

TEST_F(FLayerRandomTest, BackPropRandom)
{
    TensorGPUFloat de_dx_prev(inputs);
	TensorFloat de_dx_prev_host(inputsHost);

	FLayerCudaFTS flayer(weights, biases, true);
	FLayerFTS flayer_host(weights, biases, true);

	flayer.PrepareForTraining();
	flayer_host.PrepareForTraining();

	flayer.set_de_dx(outputs);
	flayer_host.set_de_dx(outputsHost);

	flayer.BackPropagate(inputs, de_dx_prev);
	flayer_host.BackPropagate(inputsHost, de_dx_prev_host);
    
	TensorFloat de_dx_prev_gpu = de_dx_prev;
	for(unsigned i = 0; i < de_dx_prev_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_dx_prev_host[i], de_dx_prev_gpu[i], 100.*std::numeric_limits<float>::epsilon());
	}

	TensorFloat de_dw_host = flayer_host.de_dw();
	TensorFloat de_dw = flayer.de_dw();
	for(unsigned i = 0; i < de_dw_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_dw_host[i], de_dw[i], 100.*std::numeric_limits<float>::epsilon());
	}

	TensorFloat de_dp_host = flayer_host.de_db();
	TensorFloat de_db = flayer.de_db();
	for(unsigned i = 0; i < de_dp_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_dp_host[i], de_db[i], 100.*std::numeric_limits<float>::epsilon());
	}
}

TEST_F(FLayerRandomTest, ComputeDerivRandom)
{	
	FLayerCudaFTS flayer(weights, biases, true);
	FLayerFTS flayer_host(weights, biases, true);

	flayer.PrepareForTraining();
	flayer_host.PrepareForTraining();

	flayer.set_de_dx(outputs);
	flayer_host.set_de_dx(outputsHost);

	flayer.ComputeGradient(inputs);  
	flayer_host.ComputeGradient(inputsHost);
	
	TensorFloat de_dw_host = flayer_host.de_dw();
	TensorFloat de_dw = flayer.de_dw();
	for(unsigned i = 0; i < de_dw_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_dw_host[i], de_dw[i], 100.*std::numeric_limits<float>::epsilon());
	}

	TensorFloat de_dp_host = flayer_host.de_db();
	TensorFloat de_db = flayer.de_db();
	for(unsigned i = 0; i < de_dp_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_dp_host[i], de_db[i], 100.*std::numeric_limits<float>::epsilon());
	}
}


class FLayerNumericTest : public CUDATest {
protected:
	TensorDouble inputs, weights, biases, outputs;
    TensorDouble e_host, dedx_prev;
	virtual void SetUp()
	{
		CUDATest::SetUp();

        inputs = TensorDouble(182, 1, 1);		
		outputs = TensorDouble(84, 1, 1);
        dedx_prev = TensorDouble(inputs.dims());

		std::vector<UINT> wdims;
		wdims.push_back(84);
		wdims.push_back(182);
		weights = TensorDouble(wdims);

		std::vector<UINT> tfdims;
		tfdims.push_back(1); // tansigmod num = 1 
		tfdims.push_back(84);
		biases = TensorDouble(tfdims);  // tansigmod num = 1


		for (unsigned i = 0; i < inputs.num_elements(); i++) {
			inputs[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < biases.num_elements(); i++) {
			biases[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < outputs.num_elements(); i++) {
			outputs[i] = float(std::rand()) / RAND_MAX;
		}
	}

    MSEFunction<double> performance_function;
	virtual void TearDown()
	{
	}

};

TEST_F(FLayerNumericTest, ComputeDerivNumeric)
{
    //Finite differences epsilon
    const double eps = 0.00001;
    //Number of random weights choise and test 
    const int test_iterations = 5;

	FLayerDTS flayer(weights, biases, true);
    flayer.InitWeights(&RandomWeightInit);
    flayer.PrepareForTraining();
    flayer.Propagate(inputs);
    e_host = performance_function.dydx(flayer.out() - outputs);
    flayer.set_de_dx(e_host);
    //flayer.ComputeGradient(inputs);
    flayer.BackPropagate(inputs,dedx_prev);
	
	TensorDouble de_dw_host = flayer.de_dw();
    //=============== Weights testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int nweights = flayer.weights().num_elements();
        int weight_idx = int((nweights - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double grad_comp = flayer.de_dw()[weight_idx];
        TensorDouble* weights_mutable = const_cast<TensorDouble*>(&flayer.weights());
        (*weights_mutable)[weight_idx] -= eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_1 = performance_function(e_host);
        (*weights_mutable)[weight_idx] += 2*eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_2 = performance_function(e_host);

        double numeric_grad = (loss_2 - loss_1)/(2*eps);
        //Get back weight 
        (*weights_mutable)[weight_idx] -= eps;
        EXPECT_NEAR(numeric_grad, grad_comp, 10*eps)<<"Weight gradient for flayer don't match to numeric"<<std::endl;
    }

    //=============== Transfer function parameters testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int nparams = flayer.biases().num_elements();
        int param_idx = int((nparams - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double grad_comp = flayer.de_db()[param_idx];
        TensorDouble* params_mutable = const_cast<TensorDouble*>(&flayer.biases());
        (*params_mutable)[param_idx] -= eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_1 = performance_function(e_host);
        (*params_mutable)[param_idx] += 2*eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_2 = performance_function(e_host);

        double numeric_grad = (loss_2 - loss_1)/(2*eps);
        //Get back weight 
        (*params_mutable)[param_idx] -= eps;
        EXPECT_NEAR(numeric_grad, grad_comp, 10*eps)<<"Parameter gradient for flayer don't match to numeric"<<std::endl;
    }

    //=============== Derrivative with respect to inputs testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int ninputs = inputs.num_elements();
        int input_idx = int((ninputs - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double grad_comp = dedx_prev[input_idx];
        inputs[input_idx] -= eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_1 = performance_function(e_host);
        inputs[input_idx] += 2*eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_2 = performance_function(e_host);

        double numeric_grad = (loss_2 - loss_1)/(2*eps);
        //Get back weight 
        inputs[input_idx] -= eps;
        EXPECT_NEAR(numeric_grad, grad_comp, 10*eps)<<"Parameter gradient for flayer don't match to numeric"<<std::endl;
    }

}



TEST_F(FLayerNumericTest, DISABLED_ComputeSecondDerivNumeric)
{
    //Finite differences epsilon
    const double eps = 0.00001;
    //Number of random weights choise and test 
    const int test_iterations = 5;
    //Alias
    TensorDouble& d2edx2_prev = dedx_prev;

	FLayerDTS flayer(weights, biases, true);
    flayer.InitWeights(&RandomWeightInit);
    flayer.PrepareForTraining();
    flayer.Propagate(inputs);
    e_host = performance_function.d2ydx2(flayer.out() - outputs);
    //e_host = performance_function.dydx(flayer.out() - outputs);
    flayer.set_d2e_dx2(e_host);
    //flayer.ComputeGradient(inputs);
    flayer.BackPropagateHessian(inputs,d2edx2_prev);
	
    TensorDouble d2e_dw2_host = flayer.d2e_dw2();
    //=============== Weights testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int nweights = flayer.weights().num_elements();
        int weight_idx = int((nweights - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double hess_comp = flayer.d2e_dw2()[weight_idx];
        //Compute central element of finite difference
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_center = performance_function(e_host);
        //Compute left element of finite difference
        TensorDouble* weights_mutable = const_cast<TensorDouble*>(&flayer.weights());
        (*weights_mutable)[weight_idx] -= eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_m_eps = performance_function(e_host); 
        //Compute right element of finite difference
        (*weights_mutable)[weight_idx] += 2*eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_p_eps = performance_function(e_host);

        double numeric_hess = (loss_p_eps - 2*loss_center + loss_m_eps)/(eps*eps);
        //Get back weight 
        (*weights_mutable)[weight_idx] -= eps;
        //EXPECT_NEAR(numeric_hess, hess_comp, 10*eps)<<"Weight gradient for flayer don't match to numeric"<<std::endl;
        EXPECT_PRED2(Predicates::SameOrderAndSign,numeric_hess, hess_comp)<<"Weight hessian for flayer don't match to numeric"<<std::endl;
    }

    ////=============== Transfer function parameters testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int nparams = flayer.biases().num_elements();
        int param_idx = int((nparams - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double hess_comp = flayer.d2e_db2()[param_idx];
        //Compute central element of finite difference
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_center = performance_function(e_host);
        //Compute left element of finite difference
        TensorDouble* params_mutable = const_cast<TensorDouble*>(&flayer.biases());
        (*params_mutable)[param_idx] -= eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_m_eps = performance_function(e_host); 
        //Compute right element of finite difference
        (*params_mutable)[param_idx] += 2*eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_p_eps = performance_function(e_host);

        double numeric_hess = (loss_p_eps - 2*loss_center + loss_m_eps)/(eps*eps);
        //Get back weight 
        (*params_mutable)[param_idx] -= eps;
        //EXPECT_NEAR(numeric_hess, hess_comp, 10*eps)<<"Transfer function param gradient for flayer don't match to numeric"<<std::endl;
        EXPECT_PRED2(Predicates::SameOrderAndSign,numeric_hess, hess_comp)<<"Transfer function param hessian for flayer don't match to numeric"<<std::endl;
    }

    ////=============== Derrivative with respect to inputs testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int ninputs = inputs.num_elements();
        int inp_idx = int((ninputs - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double hess_comp = d2edx2_prev[inp_idx];
        //Compute central element of finite difference
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_center = performance_function(e_host);
        //Compute left element of finite difference
        inputs[inp_idx] -= eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_m_eps = performance_function(e_host); 
        //Compute right element of finite difference
        inputs[inp_idx] += 2*eps;
        flayer.Propagate(inputs);
        e_host = flayer.out() - outputs;
        double loss_p_eps = performance_function(e_host);

        double numeric_hess = (loss_p_eps - 2*loss_center + loss_m_eps)/(eps*eps);
        //Get back weight 
        inputs[inp_idx] -= eps;
        //EXPECT_NEAR(numeric_hess, hess_comp, 10*eps)<<"Transfer function param gradient for flayer don't match to numeric"<<std::endl;    
        EXPECT_PRED2(Predicates::SameOrderAndSign,numeric_hess, hess_comp)<<"Input hessian for flayer don't match to numeric"<<std::endl;
    }

}