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
class PLayerPropagateTest : public CUDATest {
protected:
	TensorGPUFloat inputs, weights, biases;
	virtual void SetUp()
	{
		CUDATest::SetUp();
	}
	virtual void TearDown()
	{

	}
};

TEST_F(PLayerPropagateTest, PropagateOnes)
{
	//Zero inputs, weights, biases, connections
	inputs = TensorFloat(40, 40, 5);
	weights = TensorFloat(1,1,5);
	biases = TensorFloat(1,1,5);
	TensorFloat inputsHost(inputs.w(),inputs.h(), inputs.d());
	for (unsigned i = 0; i < inputsHost.num_elements(); ++i)	{
		inputsHost[i] = 1;
	}
	TensorFloat weightsHost(weights.w(),weights.h(), weights.d());
	for (unsigned i = 0; i < weightsHost.num_elements(); ++i)	{
		weightsHost[i] = 1;
	}

	TensorFloat biasesHost(biases.w(),biases.h(), biases.d());
	for (unsigned i = 0; i < biasesHost.num_elements(); ++i)	{
		biasesHost[i] = 0;
	}
	inputs = inputsHost;
	weights = weightsHost;
	biases = biasesHost;
	PoolingLayerT<TensorGPU, float>::Params params_gpu;
	params_gpu.inp_width = inputs.w();
	params_gpu.inp_height = inputs.h();
	params_gpu.ninputs = 5;
	params_gpu.sx = 2;
	params_gpu.sy = 2;
    params_gpu.pooling_type = PoolingLayerT<TensorGPU, float>::eAverage;
	PoolingLayer<TensorGPU, float> player_gpu(params_gpu);	
	player_gpu.Propagate(inputs);

	PoolingLayerT<Tensor, float>::Params params_host;
	params_host.inp_width = inputs.w();
	params_host.inp_height = inputs.h();
	params_host.ninputs = 5;
	params_host.sx = 2;
	params_host.sy = 2;
    params_host.pooling_type = PoolingLayerT<Tensor, float>::eAverage;

	PoolingLayer<Tensor, float> player_host(params_host);
	player_host.Propagate(inputsHost);
	AssertNearTensors(player_host.out(), player_gpu.out(), "Error in slayer random propagation");

}

TEST_F(PLayerPropagateTest, PropagateRandomSimple)
{
	//Zero inputs, weights, biases, connections
	inputs = TensorFloat(40, 40, 5);
	weights = TensorFloat(1,1,5);
	biases = TensorFloat(1,1,5);
	TensorFloat inputsHost(inputs.w(),inputs.h(), inputs.d());
	for (unsigned i = 0; i < inputsHost.num_elements(); ++i)	{
		inputsHost[i] = float(rand())/RAND_MAX;
	}
	TensorFloat weightsHost(weights.w(),weights.h(), weights.d());
	for (unsigned i = 0; i < weightsHost.num_elements(); ++i)	{
		weightsHost[i] = float(rand())/RAND_MAX;
	}

	TensorFloat biasesHost(biases.w(),biases.h(), biases.d());
	for (unsigned i = 0; i < biasesHost.num_elements(); ++i)	{
		biasesHost[i] = float(rand())/RAND_MAX;
	}

	inputs = inputsHost;
	weights = weightsHost;
	biases = biasesHost;
	PoolingLayerT<TensorGPU, float>::Params params_gpu;
	params_gpu.inp_width = inputs.w();
	params_gpu.inp_height = inputs.h();
	params_gpu.ninputs = 5;
	params_gpu.sx = 2;
	params_gpu.sy = 2;
    params_gpu.pooling_type = PoolingLayerT<TensorGPU, float>::eAverage;

	PoolingLayer<TensorGPU, float> player_gpu(params_gpu);	
	player_gpu.Propagate(inputs);

	PoolingLayerT<Tensor, float>::Params params_host;
	params_host.inp_width = inputs.w();
	params_host.inp_height = inputs.h();
	params_host.ninputs = 5;
	params_host.sx = 2;
	params_host.sy = 2;
    params_host.pooling_type = PoolingLayerT<Tensor, float>::eAverage;

	PoolingLayer<Tensor, float> player_host(params_host);
	player_host.Propagate(inputsHost);
	AssertNearTensors(player_host.out(), player_gpu.out(), "Error in slayer random propagation");

}


TEST_F(PLayerPropagateTest, PropagateRandomIrregularAverage)
{
	//Zero inputs, weights, biases, connections
	inputs = TensorFloat(40, 12, 5);
	TensorFloat inputsHost(inputs.w(),inputs.h(), inputs.d());
	for (unsigned i = 0; i < inputsHost.num_elements(); ++i)	{
		inputsHost[i] = float(rand())/RAND_MAX;
	}
	TensorFloat weightsHost(weights.w(),weights.h(), weights.d());
	for (unsigned i = 0; i < weightsHost.num_elements(); ++i)	{
		weightsHost[i] = float(rand())/RAND_MAX;
	}

	TensorFloat biasesHost(biases.w(),biases.h(), biases.d());
	for (unsigned i = 0; i < biasesHost.num_elements(); ++i)	{
		biasesHost[i] = float(rand())/RAND_MAX;
	}
	inputs = inputsHost;
	weights = weightsHost;
	biases = biasesHost;
	PoolingLayerT<TensorGPU, float>::Params params_gpu;
	params_gpu.inp_width = inputs.w();
	params_gpu.inp_height = inputs.h();
	params_gpu.ninputs = 5;
	params_gpu.sx = 3;
	params_gpu.sy = 1;
    params_gpu.pooling_type = PoolingLayerT<TensorGPU, float>::eAverage;


	PoolingLayer<TensorGPU, float> player_gpu(params_gpu);	
	player_gpu.Propagate(inputs);

	PoolingLayerT<Tensor, float>::Params params_host;
	params_host.inp_width = inputs.w();
	params_host.inp_height = inputs.h();
	params_host.ninputs = 5;
	params_host.sx = 3;
	params_host.sy = 1;
    params_host.pooling_type = PoolingLayerT<Tensor, float>::eAverage;

	PoolingLayer<Tensor, float> player_host(params_host);
	player_host.Propagate(inputsHost);
	AssertNearTensors(player_host.out(), player_gpu.out(), "Error in slayer random propagation");
}

TEST_F(PLayerPropagateTest, PropagateRandomIrregularMax)
{
    //Zero inputs, weights, biases, connections
    inputs = TensorFloat(40, 12, 5);
    TensorFloat inputsHost(inputs.w(),inputs.h(), inputs.d());
    for (unsigned i = 0; i < inputsHost.num_elements(); ++i)	{
        inputsHost[i] = float(rand())/RAND_MAX;
    }
    TensorFloat weightsHost(weights.w(),weights.h(), weights.d());
    for (unsigned i = 0; i < weightsHost.num_elements(); ++i)	{
        weightsHost[i] = float(rand())/RAND_MAX;
    }

    TensorFloat biasesHost(biases.w(),biases.h(), biases.d());
    for (unsigned i = 0; i < biasesHost.num_elements(); ++i)	{
        biasesHost[i] = float(rand())/RAND_MAX;
    }
    inputs = inputsHost;
    weights = weightsHost;
    biases = biasesHost;
    PoolingLayerT<TensorGPU, float>::Params params_gpu;
    params_gpu.inp_width = inputs.w();
    params_gpu.inp_height = inputs.h();
    params_gpu.ninputs = 5;
    params_gpu.sx = 3;
    params_gpu.sy = 2;
    params_gpu.pooling_type = PoolingLayerT<TensorGPU, float>::eMax;

    PoolingLayer<TensorGPU, float> player_gpu(params_gpu);	
    player_gpu.Propagate(inputs);

    PoolingLayerT<Tensor, float>::Params params_host;
    params_host.inp_width = inputs.w();
    params_host.inp_height = inputs.h();
    params_host.ninputs = 5;
    params_host.sx = 3;
    params_host.sy = 2;
    params_host.pooling_type = PoolingLayerT<Tensor, float>::eMax;

    PoolingLayer<Tensor, float> player_host(params_host);
    player_host.Propagate(inputsHost);
    AssertNearTensors(player_host.out(), player_gpu.out(), "Error in slayer random propagation");
}

#endif //HAVE_CUDA

//Only work in debug mode and in VS
#ifdef _MSC_VER
#ifdef _DEBUG
TEST_F(PLayerPropagateTest, TestMemoryLeak)
{
    PoolingLayerT<Tensor, float>::Params params;
    params.inp_width = 32;
    params.inp_height = 28;
    params.ninputs = 5;
    params.sx = 3;
    params.sy = 2;
    params.pooling_type = PoolingLayerT<Tensor, float>::eMax;

    _CrtMemState s1, s2, s3;
    _CrtMemCheckpoint( &s1 );         
    for (int i = 0; i < 10; ++i)  {
        PoolingLayer<Tensor, float> player(params);	
    }
    _CrtMemCheckpoint( &s2 );
    _CrtMemDifference( &s3, &s1, &s2);
    ASSERT_EQ(s3.lCounts[0] | s3.lCounts[1] | s3.lCounts[2] | s3.lCounts[3] |
        s3.lSizes[0]  | s3.lSizes[1]  | s3.lSizes[2]  | s3.lSizes[3] , 0)<<"Memory leaks in Tensor"<<std::endl;
}
#endif
#endif