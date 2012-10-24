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

#include "precomp.h"

#ifdef HAVE_CUDA
class CLayerTest : public CUDATest {
protected:
	TensorGPUFloat weights, biases, inputs, outputs;
	TensorGPUInt conn_map;
	TensorFloat inputsHost, weightsHost, biasesHost, outputsHost;
	TensorInt connHost;
	virtual void SetUp()
	{
		CUDATest::SetUp();

		inputs = TensorGPUFloat(20+(7-1), 20+(7-1), 3);		

		std::vector<UINT> wdims;
		wdims.push_back(7); 
		wdims.push_back(7);
		wdims.push_back(3); 
		wdims.push_back(4);
		weights = TensorGPUFloat(wdims);

		std::vector<UINT> bdims;
		bdims.push_back(1); 
		bdims.push_back(4);
		biases = TensorGPUFloat(bdims);  

		std::vector<UINT> dims(2);
		dims[0] = inputs.d();
		dims[1] = weights.d2();
		conn_map = TensorGPUInt(dims);

		outputs = TensorGPUFloat(inputs.w()-weights.w()+1,inputs.h()-weights.h()+1, weights.d2());

		inputsHost = TensorFloat(inputs.dims());
		weightsHost = TensorFloat(weights.dims());
		biasesHost = TensorFloat(biases.dims());
		outputsHost = TensorFloat(outputs.dims());

		connHost = TensorInt(conn_map.dims());
	}

    virtual void InitZeros()
    {
        for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
			inputsHost[i] = 0.f;
		}
		for (unsigned i = 0; i < weightsHost.num_elements(); i++) {
			weightsHost[i] = 1.f;
		}
		for (unsigned i = 0; i < biasesHost.num_elements(); i++) {
			biasesHost[i] = 0.f;
		}
		for (unsigned i = 0; i < outputsHost.num_elements(); i++) {
			outputsHost[i] = 0.f;
		}
		for (unsigned i = 0; i < connHost.num_elements(); i++) {
			connHost[i] = 1;
		}

		inputs = inputsHost;
		weights = weightsHost;
		biases = biasesHost;
		outputs = outputsHost;
        conn_map = connHost;
    }
    virtual void InitOnes()
    {
        for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
			inputsHost[i] = 1.f;
		}
		for (unsigned i = 0; i < weightsHost.num_elements(); i++) {
			weightsHost[i] = 1.f;
		}
		for (unsigned i = 0; i < biasesHost.num_elements(); i++) {
			biasesHost[i] = 0.0f;
		}
		for (unsigned i = 0; i < outputsHost.num_elements(); i++) {
			outputsHost[i] = 0.f;
		}
		for (unsigned i = 0; i < connHost.num_elements(); i++) {
			connHost[i] = 1;
		}

		inputs = inputsHost;
		weights = weightsHost;
		biases = biasesHost;
		outputs = outputsHost;
        conn_map = connHost;
    }
    virtual void InitRandom()
    {       

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
			outputsHost[i] = 0.f;
		}
		for (unsigned i = 0; i < connHost.num_elements(); i++) {
			connHost[i] = std::rand() % 2;
		}

		inputs = inputsHost;
		weights = weightsHost;
		biases = biasesHost;
		outputs = outputsHost;
        conn_map = connHost;
    }
    virtual void InitPattern()
    {
        for (unsigned i = 0; i < inputsHost.num_elements(); i++) {
			//inputsHost[i] = float(i) / inputsHost.num_elements();
            inputsHost[i] = float(i % 100);
		}

		for (unsigned i = 0; i < weightsHost.num_elements(); i++) {
			//weightsHost[i] = float(i) / inputsHost.num_elements();
            weightsHost[i] = float(i % 100);
		}
		for (unsigned i = 0; i < biasesHost.num_elements(); i++) {
			//biasesHost[i] = float(i) / inputsHost.num_elements();
            biasesHost[i] = float(i % 100);
		}
		for (unsigned i = 0; i < outputsHost.num_elements(); i++) {
			outputsHost[i] = 0.f;
		}
		for (unsigned i = 0; i < connHost.num_elements(); i++) {
			connHost[i] = i%2;
            //connHost[i] = 1;
		}

		inputs = inputsHost;
		weights = weightsHost;
		biases = biasesHost;
		outputs = outputsHost;
        conn_map = connHost;
    }

	virtual void TearDown()
	{
	}
};

TEST_F(CLayerTest, PropagateZeros)
{
    InitZeros();
	cudacnn::CLayer<cudacnn::TensorGPU, float, cudacnn::TansigMod<float> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);

	clayer.Propagate(inputs);
	TensorFloat outHost;
	outHost = clayer.out();

	for (unsigned i = 0; i < outHost.num_elements(); i++) {
		ASSERT_EQ(outHost[i], 0);
	}
}

TEST_F(CLayerTest, BackPropZeros)
{
    InitZeros();

	TensorGPUFloat de_dx_prev(inputs);

	cudacnn::CLayer<cudacnn::TensorGPU, float, cudacnn::TansigMod<float> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);
   	clayer.PrepareForTraining();
	clayer.set_de_dx(outputs);
	clayer.BackPropagate(inputs, de_dx_prev);

    TensorFloat de_dx_prev_host = de_dx_prev;
	for(unsigned i = 0; i < de_dx_prev_host.num_elements(); ++i){
		ASSERT_NEAR(de_dx_prev_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
	TensorFloat de_dw_host = clayer.de_dw();
	for(unsigned i = 0; i < de_dw_host.num_elements(); ++i){
		ASSERT_NEAR(de_dw_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
	TensorFloat de_db_host = clayer.de_db();
	for(unsigned i = 0; i < de_db_host.num_elements(); ++i){
		ASSERT_NEAR(de_db_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
}

TEST_F(CLayerTest, ComputeDerivZeros)
{
    InitZeros();
	cudacnn::CLayer<cudacnn::TensorGPU, float, cudacnn::TansigMod<float> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);    
	clayer.set_de_dx(outputs);
	clayer.PrepareForTraining();
	clayer.ComputeGradient(inputs);

	TensorFloat de_dw_host = clayer.de_dw();
	for(unsigned i = 0; i < de_dw_host.num_elements(); ++i){
		ASSERT_NEAR(de_dw_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
	TensorFloat de_db_host = clayer.de_db();
	for(unsigned i = 0; i < de_db_host.num_elements(); ++i){
		ASSERT_NEAR(de_db_host[i], 0.f, std::numeric_limits<float>::epsilon());
	}
}

TEST_F(CLayerTest, PropagateOnes)
{
    InitOnes();
	cudacnn::CLayer<cudacnn::TensorGPU, float, cudacnn::TansigMod<float> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);
	clayer.Propagate(inputs);

	TensorFloat out_cuda_host, ws_cuda_host;
	out_cuda_host = clayer.out();
	cudacnn::TansigMod<float> transfer_func;

	for (unsigned i = 0; i < out_cuda_host.num_elements(); i++) 
	{
		float true_ws = float(weightsHost.w()*weightsHost.h()*inputs.d());
		float true_out = transfer_func(true_ws);
		ASSERT_EQ(out_cuda_host[i], true_out)<< "outHost is wrong at index " << i;
	}
}

class CLayerRandomTest : public CLayerTest {
protected:
    MSEFunction<double> performance_function;
	virtual void SetUp()
	{	
		CLayerTest::SetUp();

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
		for (unsigned i = 0; i < connHost.num_elements(); i++) {
			connHost[i] = std::rand() % 2;
		}

		inputs = inputsHost;
		weights = weightsHost;
		biases = biasesHost;
		outputs = outputsHost;

		conn_map = connHost;	
	}
	virtual void TearDown()
	{
	}

};

TEST_F(CLayerRandomTest, PropagateRandom)
{
	cudacnn::CLayer<cudacnn::TensorGPU, float, cudacnn::TansigMod<float> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);
	cudacnn::CLayer<cudacnn::Tensor, float, cudacnn::TansigMod<float> > clayer_host(inputs.w(), inputs.h(), true, weights, biases, conn_map);
	
	clayer.Propagate(inputs);
	clayer_host.Propagate(inputsHost);

	TensorFloat outHost = clayer.out();
	TensorFloat true_out = clayer_host.out();

	for (unsigned i = 0; i < outHost.num_elements(); i++) 
	{
		ASSERT_NEAR(outHost[i], true_out[i], 1000.*FLT_EPSILON)<< "outHost is wrong at index " << i;
	}
}

TEST_F(CLayerRandomTest, BackPropRandom)
{
    TensorGPUFloat de_dx_prev(inputs);
	TensorFloat de_dx_prev_host(inputsHost);

	cudacnn::CLayer<cudacnn::TensorGPU, float, cudacnn ::TansigMod<float> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);
	cudacnn::CLayer<cudacnn::Tensor, float, cudacnn::TansigMod<float> > clayer_host(inputs.w(), inputs.h(), true, weights, biases, conn_map);

	clayer.PrepareForTraining();
	clayer_host.PrepareForTraining();

	clayer.set_de_dx(outputs);
	clayer_host.set_de_dx(outputsHost);

	clayer.BackPropagate(inputs, de_dx_prev);
	clayer_host.BackPropagate(inputsHost, de_dx_prev_host);
    
	TensorFloat de_dx_prev_gpu = de_dx_prev;
	for(unsigned i = 0; i < de_dx_prev_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_dx_prev_host[i], de_dx_prev_gpu[i], 1000.*std::numeric_limits<float>::epsilon());
	}

	TensorFloat de_dw_host = clayer_host.de_dw();
	TensorFloat de_dw = clayer.de_dw();
	for(unsigned i = 0; i < de_dw_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_dw_host[i], de_dw[i], 1000.*std::numeric_limits<float>::epsilon());
	}

	TensorFloat de_db_host = clayer_host.de_db();
	TensorFloat de_db = clayer.de_db();
	for(unsigned i = 0; i < de_db_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_db_host[i], de_db[i], 1000.*std::numeric_limits<float>::epsilon());
	}
}

TEST_F(CLayerRandomTest, ComputeDerivRandom)
{	
	cudacnn::CLayer<cudacnn::TensorGPU, float, cudacnn::TansigMod<float> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);
	cudacnn::CLayer<cudacnn::Tensor, float, cudacnn::TansigMod<float> > clayer_host(inputs.w(), inputs.h(), true, weights, biases, conn_map);

	clayer.PrepareForTraining();
	clayer_host.PrepareForTraining();

	clayer.set_de_dx(outputs);
	clayer_host.set_de_dx(outputsHost);

	clayer.ComputeGradient(inputs);  
	clayer_host.ComputeGradient(inputsHost);
	
	TensorFloat de_dw_host = clayer_host.de_dw();
	TensorFloat de_dw = clayer.de_dw();
	for(unsigned i = 0; i < de_dw_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_dw_host[i], de_dw[i], 1000.*std::numeric_limits<float>::epsilon());
	}

	TensorFloat de_db_host = clayer_host.de_db();
	TensorFloat de_db = clayer.de_db();
	for(unsigned i = 0; i < de_db_host.num_elements(); ++i)
	{
		ASSERT_NEAR(de_db_host[i], de_db[i], 1000.*std::numeric_limits<float>::epsilon());
	}
}


class CLayerNumericTest : public CUDATest {
protected:
	TensorDouble inputs, weights, biases, outputs;
	TensorInt conn_map;
    TensorDouble e_host, dedx_prev;
    MSEFunction<double> performance_function;
	virtual void SetUp()
	{
		CUDATest::SetUp();

        inputs = TensorDouble(20+(7-1), 20+(7-1), 3);		

		std::vector<UINT> wdims;
		wdims.push_back(7); 
		wdims.push_back(7);
		wdims.push_back(3); 
		wdims.push_back(4);
		weights = TensorDouble(wdims);

		std::vector<UINT> bdims;
		bdims.push_back(1); 
		bdims.push_back(4);
		biases = TensorDouble(bdims);  

		std::vector<UINT> dims(2);
		dims[0] = inputs.d();
		dims[1] = weights.d2();
		conn_map = TensorInt(dims);

		outputs = TensorDouble(inputs.w()-weights.w()+1,inputs.h()-weights.h()+1, weights.d2());
        
        dedx_prev = TensorDouble(inputs.dims());

		for (unsigned i = 0; i < inputs.num_elements(); i++) {
			inputs[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < weights.num_elements(); i++) {
			weights[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < biases.num_elements(); i++) {
			biases[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < outputs.num_elements(); i++) {
			outputs[i] = float(std::rand()) / RAND_MAX;
		}
		for (unsigned i = 0; i < conn_map.num_elements(); i++) {
			conn_map[i] = std::rand() % 2;
		}

	}


	virtual void TearDown()
	{
	}

};

TEST_F(CLayerNumericTest, ComputeDerivNumeric)
{
    //Finite differences epsilon
    const double eps = 0.00001;
    //Number of random weights choise and test 
    const int test_iterations = 15;

	cudacnn::CLayer<cudacnn::Tensor, double, cudacnn::TansigMod<double> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);
    
    clayer.PrepareForTraining();
    clayer.Propagate(inputs);
    e_host = performance_function.dydx(clayer.out() - outputs);
    clayer.set_de_dx(e_host);
    //clayer.ComputeGradient(inputs);
    clayer.BackPropagate(inputs,dedx_prev);
	
	TensorDouble de_dw_host = clayer.de_dw();
    //=============== Weights testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int nweights = clayer.weights().num_elements();
        int weight_idx = int((nweights - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double grad_comp = clayer.de_dw()[weight_idx];
        TensorDouble* weights_mutable = const_cast<TensorDouble*>(&clayer.weights());
        (*weights_mutable)[weight_idx] -= eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_1 = performance_function(e_host);
        (*weights_mutable)[weight_idx] += 2*eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_2 = performance_function(e_host);

        double numeric_grad = (loss_2 - loss_1)/(2*eps);
        //Get back weight 
        (*weights_mutable)[weight_idx] -= eps;
        //EXPECT_NEAR(numeric_grad, grad_comp, abs(0.1*numeric_grad))<<"Weight gradient for "<<i<<" layer don't match to numeric"<<std::endl;
        EXPECT_NEAR(numeric_grad, grad_comp, 10*eps)<<"Weight gradient for clayer don't match to numeric"<<std::endl;
    }

    //=============== Transfer function parameters testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int nparams = clayer.biases().num_elements();
        int param_idx = int((nparams - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double grad_comp = clayer.de_db()[param_idx];
        TensorDouble* params_mutable = const_cast<TensorDouble*>(&clayer.biases());
        (*params_mutable)[param_idx] -= eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_1 = performance_function(e_host);
        (*params_mutable)[param_idx] += 2*eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_2 = performance_function(e_host);

        double numeric_grad = (loss_2 - loss_1)/(2*eps);
        //Get back weight 
        (*params_mutable)[param_idx] -= eps;
        EXPECT_NEAR(numeric_grad, grad_comp, 10*eps)<<"Parameter gradient for clayer don't match to numeric"<<std::endl;
    }

    //=============== Derrivative with respect to inputs testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int ninputs = inputs.num_elements();
        int input_idx = int((ninputs - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double grad_comp = dedx_prev[input_idx];
        inputs[input_idx] -= eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_1 = performance_function(e_host);
        inputs[input_idx] += 2*eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_2 = performance_function(e_host);

        double numeric_grad = (loss_2 - loss_1)/(2*eps);
        //Get back weight 
        inputs[input_idx] -= eps;
        EXPECT_NEAR(numeric_grad, grad_comp, 10*eps)<<"Parameter gradient for clayer don't match to numeric"<<std::endl;
    }

}


TEST_F(CLayerNumericTest, DISABLED_ComputeSecondDerivNumeric)
{
    //Finite differences epsilon
    const double eps = 0.00001;
    //Number of random weights choise and test 
    const int test_iterations = 15;
    //Alias
    TensorDouble& d2edx2_prev = dedx_prev;

	cudacnn::CLayer<cudacnn::Tensor, double, cudacnn::TansigMod<double> > clayer(inputs.w(), inputs.h(), true, weights, biases, conn_map);
    
    clayer.PrepareForTraining();
    clayer.Propagate(inputs);
    e_host = performance_function.d2ydx2(clayer.out() - outputs);
    //e_host = performance_function.dydx(clayer.out() - outputs);
    clayer.set_d2e_dx2(e_host);
    //clayer.ComputeGradient(inputs);
    clayer.BackPropagateHessian(inputs,d2edx2_prev);
	
    TensorDouble d2e_dw2_host = clayer.d2e_dw2();
    //=============== Weights testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int nweights = clayer.weights().num_elements();
        int weight_idx = int((nweights - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double hess_comp = clayer.d2e_dw2()[weight_idx];
        //Compute central element of finite difference
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_center = performance_function(e_host);
        //Compute left element of finite difference
        TensorDouble* weights_mutable = const_cast<TensorDouble*>(&clayer.weights());
        (*weights_mutable)[weight_idx] -= eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_m_eps = performance_function(e_host); 
        //Compute right element of finite difference
        (*weights_mutable)[weight_idx] += 2*eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_p_eps = performance_function(e_host);

        double numeric_hess = (loss_p_eps - 2*loss_center + loss_m_eps)/(eps*eps);
        //Get back weight 
        (*weights_mutable)[weight_idx] -= eps;
        //Second derrivative is just an approximation, so check only the order and sign
        //EXPECT_NEAR(numeric_hess, hess_comp, 10*eps)<<"Weight gradient for clayer don't match to numeric"<<std::endl;
        EXPECT_PRED2(Predicates::SameOrderAndSign,numeric_hess, hess_comp)<<"Weight hessian for clayer don't match to numeric"<<std::endl;
    }

    //=============== Transfer function parameters testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int nparams = clayer.biases().num_elements();
        int param_idx = int((nparams - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double hess_comp = clayer.d2e_db2()[param_idx];
        //Compute central element of finite difference
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_center = performance_function(e_host);
        //Compute left element of finite difference
        TensorDouble* params_mutable = const_cast<TensorDouble*>(&clayer.biases());
        (*params_mutable)[param_idx] -= eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_m_eps = performance_function(e_host); 
        //Compute right element of finite difference
        (*params_mutable)[param_idx] += 2*eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_p_eps = performance_function(e_host);

        double numeric_hess = (loss_p_eps - 2*loss_center + loss_m_eps)/(eps*eps);
        //Get back weight 
        (*params_mutable)[param_idx] -= eps;
        //EXPECT_NEAR(numeric_hess, hess_comp, 10*eps)<<"Transfer function param gradient for clayer don't match to numeric"<<std::endl;    
        EXPECT_PRED2(Predicates::SameOrderAndSign,numeric_hess, hess_comp)<<"Parameter hessian for clayer don't match to numeric"<<std::endl;
    }

    ////=============== Derrivative with respect to inputs testing ============================
    for(int t = 0; t < test_iterations; ++t ) {
        //Pick up some random weight
        int ninputs = inputs.num_elements();
        int inp_idx = int((ninputs - 1)*(float(rand())/RAND_MAX));
        //Gradient computed by class method
        double hess_comp = d2edx2_prev[inp_idx];
        //Compute central element of finite difference
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_center = performance_function(e_host);
        //Compute left element of finite difference
        inputs[inp_idx] -= eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_m_eps = performance_function(e_host); 
        //Compute right element of finite difference
        inputs[inp_idx] += 2*eps;
        clayer.Propagate(inputs);
        e_host = clayer.out() - outputs;
        double loss_p_eps = performance_function(e_host);

        double numeric_hess = (loss_p_eps - 2*loss_center + loss_m_eps)/(eps*eps);
        //Get back weight 
        inputs[inp_idx] -= eps;
        //EXPECT_NEAR(numeric_hess, hess_comp, 10*eps)<<"Transfer function param gradient for clayer don't match to numeric"<<std::endl;    
        EXPECT_PRED2(Predicates::SameOrderAndSign,numeric_hess, hess_comp)<<"Input hessian for clayer don't match to numeric"<<std::endl;
    }

}

#endif //HAVE_CUDA